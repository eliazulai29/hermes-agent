"""
Flow command validator — the verify-before-save gate for teach-mode processes.

WHY THIS EXISTS
---------------
Teach mode lets the agent build a reusable process by demonstrating it, then
saves it as a flow skill (a SKILL.md with a `metadata.flow` block). The danger:
when the agent WRITES the saved skill, it can fill in concrete commands from its
own training memory instead of from what it actually ran — e.g. inventing
`python3 .../skills/email/google-workspace/...` when the real path is
`.../skills/productivity/google-workspace/...`, or `gmail list --max-results`
when the real CLI is `gmail search --max`. A developer caught these by hand;
a real user never could. The result is a saved process that fails on its first
unattended run.

THIS GATE makes that impossible: before a flow skill is written to disk, every
concrete shell command in it is validated against the real filesystem and the
real CLIs. If any command references a path that doesn't exist or a flag the
target script doesn't accept, the save is BLOCKED and the specific failures are
returned so the agent can fix them.

WHAT IT CHECKS (per command line found in the skill body / steps)
  1. Script paths (*.py, *.sh, *.js) the command invokes → must exist on disk.
  2. Bare `python3 SCRIPT` for scripts that need engine deps → warn to use the
     engine venv interpreter (system python3 lacks google-api / etc.).
  3. Known engine CLIs (google_api.py, ...) → the subcommand + flags must parse
     against the script's real argparse (`--help`), catching wrong flags like
     `--max-results` (real: `--max`) or `--body-file` (real: `--body`).

It is deliberately CONSERVATIVE: it only blocks on things it can prove wrong
(missing file, rejected flag). Anything it can't resolve is a warning, never a
hard block — so it never falsely rejects a valid process.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# The engine's OWN interpreter — the one this validator (and any scheduled flow)
# runs under. This is the production-correct reference: no hardcoded venv path,
# no HERMES_HOME assumptions. Whatever Python is executing the engine IS the
# interpreter a cron-run flow will use, so commands should target it.
def _engine_python() -> str:
    return sys.executable

# File extensions we treat as "a script that must exist on disk".
_SCRIPT_EXTS = (".py", ".sh", ".js", ".mjs", ".ts")

# Cross-platform launcher matcher. Matches a command line that starts with a
# known interpreter/launcher OR an absolute path to one, on BOTH POSIX and
# Windows. Covers: python/python3/py, bash/sh, node; bare names, POSIX paths
# (/usr/bin/python), Windows paths (C:\Py\python.exe, with .exe), and ~/… or
# ./… script paths ending in a known extension. Anchored, case-insensitive for
# the .exe tail so 'PYTHON.EXE' also matches.
_LAUNCHER_RE = re.compile(
    r"^(?:"
    r"(?:python3?|py|bash|sh|node)(?:\.exe)?"            # bare launcher (+opt .exe)
    r"|[A-Za-z]:[\\/][^\s]*?(?:python|node|bash)[^\s]*"  # Windows path to interpreter
    r"|[\\/][^\s]*?(?:python|node|bash)[^\s]*"           # POSIX path to interpreter
    r"|[~./][^\s]*\.(?:py|sh|js|mjs|ts)"                 # ~/ or ./ script path
    r")\b",
    re.IGNORECASE,
)


@dataclass
class CommandIssue:
    command: str
    reason: str
    suggestion: str
    fatal: bool = True  # True → blocks save; False → warning only


@dataclass
class ValidationResult:
    ok: bool
    issues: List[CommandIssue] = field(default_factory=list)

    @property
    def fatal_issues(self) -> List[CommandIssue]:
        return [i for i in self.issues if i.fatal]

    @property
    def warnings(self) -> List[CommandIssue]:
        return [i for i in self.issues if not i.fatal]


def is_flow_skill(content: str) -> bool:
    """True if the SKILL.md frontmatter declares a `flow:` manifest block."""
    if not content.lstrip().startswith("---"):
        return False
    body_start = content.find("---")
    fm_end = content.find("---", body_start + 3)
    fm = content if fm_end == -1 else content[body_start + 3 : fm_end]
    # `flow:` key anywhere under metadata (matches metadata.hermes.flow and
    # metadata.flow shapes used by the builder).
    return bool(re.search(r"^\s*flow:\s*$", fm, re.MULTILINE) or re.search(r"\n\s*flow:\s*\n", fm))


def _expand(path: str) -> str:
    """Expand ~ and env vars in a path string the way a shell would."""
    return os.path.expandvars(os.path.expanduser(path))


def _is_explicit_path(p: str) -> bool:
    """True if `p` is an explicit path (not a bare PATH-resolved name) on either
    POSIX or Windows: starts with / ~ ./ ../ \\ , contains a separator, or has a
    Windows drive prefix like C:\\ ."""
    if p.startswith(("/", "~", "./", "../", "\\", ".\\", "..\\")):
        return True
    if "/" in p or "\\" in p:
        return True
    # Windows drive letter, e.g. C:\ or C:/
    if re.match(r"^[A-Za-z]:[\\/]", p):
        return True
    return False


def _extract_command_lines(content: str) -> List[str]:
    """
    Pull candidate shell-command lines out of a SKILL.md body.

    Sources, in order of reliability:
      1. Fenced code blocks (```bash / ```sh / ``` ).
      2. Inline code spans (`...`) that look like a command.
      3. Bare lines that start with a known command launcher.

    We only keep lines that reference a script/CLI we can actually check —
    pure prose is ignored.
    """
    cmds: List[str] = []

    # 1. Fenced code blocks.
    for block in re.findall(r"```[a-zA-Z0-9]*\n(.*?)```", content, re.DOTALL):
        for line in block.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                cmds.append(line)

    # 2. Inline code spans + 3. bare lines — scan line by line.
    for raw in content.splitlines():
        line = raw.strip()
        # Inline `code` spans.
        for span in re.findall(r"`([^`]+)`", raw):
            cmds.append(span.strip())
        # Bare command-looking lines (start with a launcher). Cross-platform:
        # accept POSIX (/usr/bin/python) AND Windows (C:\...\python.exe) paths,
        # and python/python3/py launchers.
        if re.match(_LAUNCHER_RE, line):
            cmds.append(line)

    # Keep only lines that look like real commands referencing a script/CLI.
    out: List[str] = []
    seen = set()
    for c in cmds:
        c = c.strip().strip("`").strip()
        if not c or c in seen:
            continue
        if not _looks_like_command(c):
            continue
        seen.add(c)
        out.append(c)
    return out


def _looks_like_command(s: str) -> bool:
    """Heuristic: does this string reference a script/interpreter we can check?"""
    if any(ext in s for ext in _SCRIPT_EXTS):
        return True
    if re.match(_LAUNCHER_RE, s):
        return True
    return False


def _safe_split(cmd: str) -> Optional[List[str]]:
    """Tokenize a command line, respecting the host OS quoting/escaping rules.

    CRITICAL cross-platform detail: POSIX shlex treats '\\' as an escape char, so
    a Windows path like C:\\Users\\x\\foo.py gets mangled to C:Usersxfoo.py. We
    split with posix=(not Windows) so backslash paths survive on Windows. On a
    POSIX host a Windows-style path is unusual anyway; we still try a backslash-
    preserving fallback if the posix split looks like it ate path separators.
    """
    is_windows = os.name == "nt"
    try:
        tokens = shlex.split(cmd, posix=not is_windows)
    except ValueError:
        return None
    # POSIX-host fallback: if a token lost its separators but the raw command
    # clearly had a Windows path (drive letter + backslashes), re-split non-posix
    # so the path token stays intact and the existence check is meaningful.
    if not is_windows and re.search(r"[A-Za-z]:\\", cmd):
        try:
            return shlex.split(cmd, posix=False)
        except ValueError:
            return tokens
    return tokens


def _script_token(tokens: List[str]) -> Optional[str]:
    """Find the script-path token (the first arg ending in a script ext)."""
    for tok in tokens:
        if tok.endswith(_SCRIPT_EXTS):
            return tok
    return None


def _validate_one(cmd: str) -> List[CommandIssue]:
    issues: List[CommandIssue] = []
    tokens = _safe_split(cmd)
    if not tokens:
        return issues  # unparseable → skip (don't false-block)

    interpreter = tokens[0]
    script = _script_token(tokens)

    # (1) Script path must exist on disk.
    if script:
        resolved = _expand(script)
        # Only check absolute / ~ / explicit-relative paths (skip bare 'foo.py'
        # that might be PATH-resolved or cwd-relative in a known dir).
        # Cross-platform: accept POSIX ('/','./','../'), '~', AND Windows paths
        # (backslash separators, or a 'C:' drive letter).
        if _is_explicit_path(script):
            if not os.path.exists(resolved):
                suggestion = _suggest_path(resolved)
                issues.append(
                    CommandIssue(
                        command=cmd,
                        reason=f"script path does not exist: {script}",
                        suggestion=suggestion,
                        fatal=True,
                    )
                )
                return issues  # can't check flags against a missing script

    # (2) GENERIC dependency check: if the command runs a Python script under an
    # interpreter OTHER than the engine's, probe whether the script's imports
    # actually resolve there. We don't hardcode any script names or packages —
    # we run the script's own `--help` under the stated interpreter and look for
    # an Import/ModuleNotFoundError. If it fails to import there but the engine
    # interpreter (which a cron flow uses) imports it fine, the flow would break
    # at runtime. Works for ANY script with ANY dependency.
    if (
        script
        and os.path.exists(_expand(script))
        and _is_python_interpreter(interpreter)
        and not _same_interpreter(interpreter, _engine_python())
    ):
        dep_issue = _check_interpreter_deps(interpreter, script, cmd)
        if dep_issue:
            issues.append(dep_issue)

    # (3) Flags must parse against the real CLI (dry-run --help arg-check).
    if script and os.path.exists(_expand(script)):
        flag_issue = _check_flags(tokens, script)
        if flag_issue:
            issues.append(flag_issue)

    return issues


def _is_python_interpreter(tok: str) -> bool:
    base = os.path.basename(tok)
    return base in ("python", "python3") or base.startswith("python")


def _same_interpreter(a: str, b: str) -> bool:
    """True if two interpreter tokens resolve to the same executable.

    Cross-platform: a path token contains either separator ('/' or '\\').
    """
    try:
        has_sep_a = ("/" in a) or ("\\" in a)
        ra = os.path.normcase(os.path.realpath(_expand(a))) if has_sep_a else a
        rb = os.path.normcase(os.path.realpath(_expand(b)))
        if ra == rb:
            return True
    except Exception:
        pass
    # A bare 'python3' could BE the engine interpreter on PATH — can't prove
    # otherwise cheaply, so treat bare names as "possibly different" (we probe).
    return False


def _check_interpreter_deps(interpreter: str, script: str, cmd: str) -> Optional[CommandIssue]:
    """Probe: does the script import cleanly under the stated interpreter?

    Runs `<interpreter> <script> --help` (no side effects — argparse prints help
    and exits) and looks for an import failure. Generic: no package/script names
    hardcoded. Only a WARNING (system python MIGHT have the deps), and suggests
    the engine interpreter that a scheduled run will actually use.
    """
    interp = _expand(interpreter)
    script_resolved = _expand(script)
    try:
        proc = subprocess.run(
            [interp, script_resolved, "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        # The interpreter itself doesn't exist → that's a real (fatal) problem.
        return CommandIssue(
            command=cmd,
            reason=f"interpreter not found: {interpreter}",
            suggestion=f"use the engine interpreter: {_engine_python()}",
            fatal=True,
        )
    except Exception:
        return None  # can't probe → don't false-block

    err = (proc.stderr or "") + (proc.stdout or "")
    if re.search(r"(ModuleNotFoundError|ImportError|No module named)", err):
        m = re.search(r"No module named ['\"]([\w.]+)['\"]", err)
        missing = f" (missing: {m.group(1)})" if m else ""
        return CommandIssue(
            command=cmd,
            reason=f"'{interpreter}' is missing a dependency this script needs{missing}",
            suggestion=f"use the engine interpreter, which has it: {_engine_python()}",
            fatal=False,  # warning — but a scheduled run WILL use the engine interp
        )
    return None


def _suggest_path(missing: str) -> str:
    """If a sibling of the missing path exists, suggest it (e.g. email→productivity)."""
    base = os.path.basename(missing)
    # Search ~/.hermes for a file with the same name.
    home = Path.home() / ".hermes"
    try:
        matches = [str(p) for p in home.rglob(base) if p.is_file()]
    except Exception:
        matches = []
    if matches:
        # Prefer the shortest / non-bundled match.
        matches.sort(key=len)
        return f"did you mean: {matches[0]}"
    return f"no file named '{base}' found under ~/.hermes"


def _check_flags(tokens: List[str], script: str) -> Optional[CommandIssue]:
    """
    Dry-run the script's argparse to see if the subcommand+flags are accepted.

    Strategy: run `<interpreter> <script> <subcmd> --help` and confirm it exits 0
    (argparse prints help + exits 0 for a VALID subcommand path). If the
    subcommand is wrong, argparse exits non-zero with 'invalid choice'. Then,
    for each `--flag` in the command, confirm it appears in that help text.
    """
    cmd_full = _expand(" ".join(tokens))
    script_resolved = _expand(script)
    # Identify subcommand tokens (positionals before the first flag, after script).
    try:
        s_idx = tokens.index(script)
    except ValueError:
        # script token may be expanded differently; match by basename
        s_idx = next((i for i, t in enumerate(tokens) if t.endswith(os.path.basename(script))), None)
        if s_idx is None:
            return None

    rest = tokens[s_idx + 1 :]
    subcmds = []
    for tok in rest:
        if tok.startswith("-"):
            break
        subcmds.append(tok)

    interp = _engine_python()
    help_argv = [interp, script_resolved, *subcmds, "--help"]
    try:
        proc = subprocess.run(
            help_argv,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return None  # can't introspect → don't false-block

    help_text = (proc.stdout or "") + (proc.stderr or "")

    # Wrong subcommand → argparse says 'invalid choice'.
    if "invalid choice" in help_text.lower():
        # Extract valid choices to suggest.
        m = re.search(r"\(choose from ([^)]+)\)", help_text)
        choices = m.group(1) if m else ""
        return CommandIssue(
            command=" ".join(tokens),
            reason=f"unknown subcommand '{' '.join(subcmds)}' for {os.path.basename(script)}",
            suggestion=f"valid subcommands: {choices}" if choices else "run with --help to see valid subcommands",
            fatal=True,
        )

    if proc.returncode != 0 and not help_text.strip():
        return None  # couldn't get help → skip

    # Each --flag used must appear in the help text.
    used_flags = [t for t in rest if t.startswith("--")]
    for flag in used_flags:
        flag_name = flag.split("=", 1)[0]
        if flag_name not in help_text:
            # Suggest a near-match flag from the help.
            avail = re.findall(r"(--[a-zA-Z][\w-]+)", help_text)
            near = _closest(flag_name, avail)
            return CommandIssue(
                command=" ".join(tokens),
                reason=f"flag '{flag_name}' is not accepted by {os.path.basename(script)} {' '.join(subcmds)}",
                suggestion=(f"did you mean '{near}'?" if near else f"valid flags: {', '.join(sorted(set(avail)))}"),
                fatal=True,
            )
    return None


def _closest(target: str, options: List[str]) -> Optional[str]:
    """Cheap closest-match (shared-prefix / substring) for a flag suggestion."""
    target_core = target.lstrip("-")
    best = None
    best_score = 0
    for opt in options:
        core = opt.lstrip("-")
        # shared prefix length
        score = 0
        for a, b in zip(target_core, core):
            if a == b:
                score += 1
            else:
                break
        if target_core in core or core in target_core:
            score += 3
        if score > best_score:
            best_score = score
            best = opt
    return best if best_score >= 2 else None


def validate_flow_content(content: str) -> ValidationResult:
    """
    Validate every concrete command in a flow SKILL.md. Returns a
    ValidationResult; `ok` is False only when there is at least one FATAL issue
    (a provably-wrong path or flag). Warnings never block the save.
    """
    if not is_flow_skill(content):
        return ValidationResult(ok=True)

    issues: List[CommandIssue] = []
    for cmd in _extract_command_lines(content):
        issues.extend(_validate_one(cmd))

    has_fatal = any(i.fatal for i in issues)
    return ValidationResult(ok=not has_fatal, issues=issues)


def format_issues_for_agent(result: ValidationResult) -> str:
    """Human/agent-readable summary of why a save was blocked."""
    lines: List[str] = []
    if result.fatal_issues:
        lines.append("Flow save BLOCKED — these commands are not valid (fix them, they won't run):")
        for i in result.fatal_issues:
            lines.append(f"  ✗ `{i.command}`")
            lines.append(f"      reason: {i.reason}")
            lines.append(f"      fix:    {i.suggestion}")
    if result.warnings:
        lines.append("Warnings (not blocking, but recommended):")
        for i in result.warnings:
            lines.append(f"  ⚠ `{i.command}` — {i.reason} → {i.suggestion}")
    return "\n".join(lines)
