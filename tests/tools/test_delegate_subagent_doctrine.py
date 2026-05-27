"""Regression tests for subagent doctrine inheritance.

Why this exists
---------------
Subagents spawned via ``delegate_task`` historically received a generic
English "You are a focused subagent" system prompt that did NOT inherit
the parent agent's SOUL.md (profile-level doctrine).  This created a
class of safety bugs for products built on Hermes that handle sensitive
data — e.g. an Israeli insurance agent (Shavit) whose main agent
operates under strict rules (Hebrew-default language, never echo
credentials, always ``vault_list`` before asking for passwords, never
cite numbers without a tool call) would silently lose ALL of those
rules when it delegated a subtask: the worker defaulted to a generic
English persona that could echo passwords back through the parent and
fabricate insurance numbers from training-data memory.

The fix exposes ``inherited_doctrine`` on ``_build_child_system_prompt``
and wires ``_build_child_agent`` to pass the parent profile's SOUL.md
through ``load_soul_md()``.  These tests pin the contract.

If a future upstream cherry-pick removes the parameter or changes the
inheritance wiring, these tests fail loudly instead of letting the
regression ship to end users.
"""

from tools.delegate_tool import _build_child_system_prompt


class TestDoctrineInheritance:
    """``inherited_doctrine`` propagates the parent's profile rules."""

    def test_doctrine_present_appears_in_prompt(self):
        doctrine = "# Test Profile\n\nשפת ברירת מחדל: עברית.\nלעולם אל תחשוף סיסמאות."
        prompt = _build_child_system_prompt(
            goal="Research migdal car insurance rates",
            inherited_doctrine=doctrine,
        )
        assert "INHERITED DOCTRINE" in prompt
        assert "שפת ברירת מחדל: עברית" in prompt
        assert "לעולם אל תחשוף סיסמאות" in prompt

    def test_doctrine_present_adds_subagent_framing(self):
        prompt = _build_child_system_prompt(
            goal="Anything",
            inherited_doctrine="Some doctrine.",
        )
        # The bridge framing tells the worker it's a subagent of a parent
        # operating under the doctrine above — critical for the worker
        # to apply the rules to its own behaviour, not just acknowledge them.
        assert "SUBAGENT" in prompt
        assert "parent agent" in prompt
        assert "doctrine above applies fully" in prompt

    def test_doctrine_present_still_includes_task(self):
        prompt = _build_child_system_prompt(
            goal="Specific task description here",
            inherited_doctrine="Doctrine X",
        )
        assert "YOUR TASK:" in prompt
        assert "Specific task description here" in prompt

    def test_doctrine_appears_before_task(self):
        """Doctrine must come BEFORE the task so the worker applies it to
        the task — not after, where the worker might miss it."""
        prompt = _build_child_system_prompt(
            goal="The task",
            inherited_doctrine="DOCTRINE_SENTINEL_ABC",
        )
        doctrine_pos = prompt.index("DOCTRINE_SENTINEL_ABC")
        task_pos = prompt.index("The task")
        assert doctrine_pos < task_pos, (
            "Inherited doctrine must precede YOUR TASK so the worker "
            "applies the rules; otherwise the LLM may treat doctrine as "
            "post-hoc reflection rather than active guidance."
        )


class TestBackwardCompatibility:
    """Callers that don't opt into doctrine inheritance still work."""

    def test_doctrine_none_falls_back_to_legacy_prompt(self):
        prompt = _build_child_system_prompt(goal="Some task")
        assert "INHERITED DOCTRINE" not in prompt
        assert "You are a focused subagent working on a specific delegated task" in prompt
        assert "YOUR TASK:" in prompt
        assert "Some task" in prompt

    def test_doctrine_empty_string_treated_as_none(self):
        prompt = _build_child_system_prompt(goal="Task", inherited_doctrine="")
        assert "INHERITED DOCTRINE" not in prompt

    def test_doctrine_whitespace_only_treated_as_none(self):
        prompt = _build_child_system_prompt(goal="Task", inherited_doctrine="   \n\t  ")
        assert "INHERITED DOCTRINE" not in prompt

    def test_orchestrator_role_still_works_with_doctrine(self):
        """Orchestrator delegation guidance must still appear when
        doctrine is also inherited — they're orthogonal concerns."""
        prompt = _build_child_system_prompt(
            goal="Decompose this",
            inherited_doctrine="Doctrine here",
            role="orchestrator",
            max_spawn_depth=2,
            child_depth=1,
        )
        assert "INHERITED DOCTRINE" in prompt
        assert "Subagent Spawning (Orchestrator Role)" in prompt
        assert "delegate_task" in prompt
