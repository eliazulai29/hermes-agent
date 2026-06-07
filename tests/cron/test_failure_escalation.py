"""Cron consecutive-failure escalation: a recurring job that keeps failing
must be auto-paused after N in a row (not fail silently forever), and a
success must reset the counter so a transient hiccup doesn't pause a healthy
job. (issue: cron silent infinite failure, 2026-06-07)"""
import os
import tempfile
import pytest


@pytest.fixture
def isolated_jobs(monkeypatch):
    monkeypatch.setenv("HERMES_HOME", tempfile.mkdtemp())
    import importlib
    from cron import jobs as J
    importlib.reload(J)
    return J


def test_auto_pause_after_threshold(isolated_jobs):
    J = isolated_jobs
    job = J.create_job(prompt="check portal", schedule="0 9 * * *", name="t")
    jid = job["id"]
    assert job["enabled"] is True

    for i in range(1, J.CRON_FAILURE_PAUSE_THRESHOLD):
        J.mark_job_run(jid, success=False, error="login changed")
        j = J.resolve_job_ref(jid)
        assert j["consecutive_failures"] == i
        assert j["enabled"] is True  # not yet paused

    # The threshold-th failure pauses it.
    J.mark_job_run(jid, success=False, error="login changed")
    j = J.resolve_job_ref(jid)
    assert j["consecutive_failures"] == J.CRON_FAILURE_PAUSE_THRESHOLD
    assert j["enabled"] is False
    assert j["state"] == "paused"
    assert j["needs_attention"] is True
    assert j["paused_reason"]


def test_success_resets_counter(isolated_jobs):
    J = isolated_jobs
    job = J.create_job(prompt="x", schedule="0 9 * * *", name="t2")
    jid = job["id"]
    J.mark_job_run(jid, success=False, error="blip")
    J.mark_job_run(jid, success=False, error="blip")
    J.mark_job_run(jid, success=True)  # recovered
    j = J.resolve_job_ref(jid)
    assert j["consecutive_failures"] == 0
    assert j["enabled"] is True
    assert j["needs_attention"] is False
