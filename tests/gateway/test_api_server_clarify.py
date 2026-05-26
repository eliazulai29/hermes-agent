"""Tests for POST /v1/runs/{run_id}/clarify and the SSE ``clarify.request``
event emitted by ``_make_clarify_callback``.

End-to-end flow (mocked agent):

  1. Client starts a run via ``POST /v1/runs``.
  2. The mocked agent invokes ``clarify_callback`` during ``run_conversation``.
  3. The api_server pushes a ``clarify.request`` event onto the run's SSE
     queue and the agent worker thread blocks on
     ``clarify_gateway.wait_for_response``.
  4. The client reads the event, then POSTs to /clarify with the answer.
  5. The blocked thread wakes up; the agent's run_conversation returns
     the answer as part of the final response.
  6. The run completes with status ``completed`` and the final response
     contains the answer.

These tests intentionally do NOT spin up a real LLM — they patch
``_create_agent`` and inject a mock that uses the real callback. That
keeps the test hermetic while still exercising the real
clarify_gateway primitive, the real SSE queue, and the real /clarify
handler.
"""

import asyncio
import json
import threading
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    return APIServerAdapter(config)


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    """aiohttp app with the /v1/runs routes registered, including /clarify."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    app.router.add_post("/v1/runs/{run_id}/clarify", adapter._handle_run_clarify)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return app


def _make_clarify_agent(
    clarify_question: str = "What's your favourite colour?",
    clarify_choices=None,
):
    """Build a mock agent whose ``run_conversation`` calls the injected
    clarify_callback once, then returns the user's answer as the final
    response (so a test can assert it round-tripped through the wire).

    The trick: ``_create_agent`` is patched to return this mock, and we
    grab the ``clarify_callback`` kwarg the api_server passed in so the
    mock's ``run_conversation`` can actually invoke it on the agent's
    worker thread.
    """
    mock_agent = MagicMock()
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0

    # Slot for the captured callback; the mock_create_agent factory
    # below populates it once the api_server actually calls _create_agent.
    callback_holder = {"cb": None}

    def _run(user_message=None, conversation_history=None, task_id=None):
        cb = callback_holder["cb"]
        assert cb is not None, "clarify_callback was not wired through _create_agent"
        answer = cb(clarify_question, clarify_choices)
        return {"final_response": f"user said: {answer}"}

    mock_agent.run_conversation.side_effect = _run
    return mock_agent, callback_holder


def _make_create_agent_factory(mock_agent, callback_holder):
    """Return a ``_create_agent`` replacement that captures
    ``clarify_callback`` into ``callback_holder`` before returning the
    mock agent.

    The api_server's ``_handle_runs`` passes the callback as a kwarg
    (see ``clarify_callback=clarify_cb`` in _run_and_close), so this
    proves the wiring is intact — not just that the code compiles.
    """
    def _factory(*args, **kwargs):
        callback_holder["cb"] = kwargs.get("clarify_callback")
        return mock_agent

    return _factory


async def _wait_for_clarify_event(adapter, run_id, timeout: float = 5.0):
    """Drain the SSE queue until a clarify.request event is seen.

    Returns the event dict, or raises asyncio.TimeoutError if the queue
    runs dry before one shows up.
    """
    q = adapter._run_streams.get(run_id)
    assert q is not None, f"no SSE queue for run {run_id}"
    deadline = asyncio.get_event_loop().time() + timeout
    seen = []
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError(
                f"clarify.request not seen in {timeout}s; saw: {seen}"
            )
        evt = await asyncio.wait_for(q.get(), timeout=remaining)
        if evt is None:
            # SSE sentinel — should not arrive before clarify
            raise AssertionError(
                f"run {run_id} finished before clarify event; saw: {seen}"
            )
        seen.append(evt.get("event"))
        if evt.get("event") == "clarify.request":
            return evt


async def _wait_for_run_status(cli, run_id, target_status: str, timeout: float = 5.0):
    """Poll GET /v1/runs/{run_id} until status matches or timeout fires."""
    import time as _t
    deadline = _t.monotonic() + timeout
    last = None
    while _t.monotonic() < deadline:
        resp = await cli.get(f"/v1/runs/{run_id}")
        assert resp.status == 200
        last = await resp.json()
        if last["status"] == target_status:
            return last
        await asyncio.sleep(0.02)
    raise AssertionError(
        f"run {run_id} never reached status {target_status} (last: {last})"
    )


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


# ---------------------------------------------------------------------------
# Happy path: agent calls clarify → SSE event → client answers → run completes
# ---------------------------------------------------------------------------


class TestClarifyHappyPath:
    @pytest.mark.asyncio
    async def test_full_clarify_roundtrip(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            mock_agent, holder = _make_clarify_agent(
                clarify_question="What's your name?",
                clarify_choices=None,
            )
            with patch.object(
                adapter,
                "_create_agent",
                side_effect=_make_create_agent_factory(mock_agent, holder),
            ):
                # Start a run.
                resp = await cli.post("/v1/runs", json={"input": "hi"})
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]

                # Drain SSE queue until the clarify.request shows up.
                evt = await _wait_for_clarify_event(adapter, run_id)
                assert evt["event"] == "clarify.request"
                assert evt["run_id"] == run_id
                assert evt["question"] == "What's your name?"
                assert evt["choices"] is None
                clarify_id = evt["clarify_id"]
                assert isinstance(clarify_id, str) and len(clarify_id) >= 6

                # POST the answer.
                ans_resp = await cli.post(
                    f"/v1/runs/{run_id}/clarify",
                    json={"clarify_id": clarify_id, "answer": "Eli"},
                )
                assert ans_resp.status == 200
                ans_body = await ans_resp.json()
                assert ans_body["object"] == "hermes.run.clarify_response"
                assert ans_body["clarify_id"] == clarify_id

                # Run should complete with the answer round-tripped.
                final = await _wait_for_run_status(cli, run_id, "completed")
                assert final["output"] == "user said: Eli"

    @pytest.mark.asyncio
    async def test_clarify_with_choices_in_event(self, adapter):
        """The choices list (if provided) must round-trip into the SSE event
        so the UI can render buttons/options correctly."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            mock_agent, holder = _make_clarify_agent(
                clarify_question="Pick one",
                clarify_choices=["red", "green", "blue"],
            )
            with patch.object(
                adapter,
                "_create_agent",
                side_effect=_make_create_agent_factory(mock_agent, holder),
            ):
                resp = await cli.post("/v1/runs", json={"input": "hi"})
                run_id = (await resp.json())["run_id"]

                evt = await _wait_for_clarify_event(adapter, run_id)
                assert evt["choices"] == ["red", "green", "blue"]
                clarify_id = evt["clarify_id"]

                await cli.post(
                    f"/v1/runs/{run_id}/clarify",
                    json={"clarify_id": clarify_id, "answer": "blue"},
                )
                final = await _wait_for_run_status(cli, run_id, "completed")
                assert final["output"] == "user said: blue"

    @pytest.mark.asyncio
    async def test_clarify_responded_event_emitted_after_post(self, adapter):
        """After /clarify resolves, a ``clarify.responded`` event should
        land on the SSE queue so the UI can dismiss its pending card
        without waiting for the next tool call."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            mock_agent, holder = _make_clarify_agent()
            with patch.object(
                adapter,
                "_create_agent",
                side_effect=_make_create_agent_factory(mock_agent, holder),
            ):
                resp = await cli.post("/v1/runs", json={"input": "hi"})
                run_id = (await resp.json())["run_id"]
                evt = await _wait_for_clarify_event(adapter, run_id)
                clarify_id = evt["clarify_id"]

                await cli.post(
                    f"/v1/runs/{run_id}/clarify",
                    json={"clarify_id": clarify_id, "answer": "purple"},
                )

                # Pull the next event(s) until we either see clarify.responded
                # or run.completed (the latter would mean we missed it).
                q = adapter._run_streams.get(run_id)
                if q is None:
                    pytest.skip("run completed before we could drain events")
                seen_responded = False
                for _ in range(20):
                    try:
                        evt = await asyncio.wait_for(q.get(), timeout=2.0)
                    except asyncio.TimeoutError:
                        break
                    if evt is None:
                        break
                    if evt.get("event") == "clarify.responded":
                        assert evt["clarify_id"] == clarify_id
                        seen_responded = True
                        break
                    if evt.get("event") == "run.completed":
                        break
                assert seen_responded, (
                    "clarify.responded event must follow a successful POST"
                )


# ---------------------------------------------------------------------------
# Error paths: bad payloads, unknown ids, wrong run
# ---------------------------------------------------------------------------


class TestClarifyErrors:
    @pytest.mark.asyncio
    async def test_unknown_run_id_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs/run_nope/clarify",
                json={"clarify_id": "x", "answer": "y"},
            )
            assert resp.status == 404
            body = await resp.json()
            assert body["error"]["code"] == "run_not_found"

    @pytest.mark.asyncio
    async def test_unknown_clarify_id_returns_409(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            mock_agent, holder = _make_clarify_agent()
            with patch.object(
                adapter,
                "_create_agent",
                side_effect=_make_create_agent_factory(mock_agent, holder),
            ):
                resp = await cli.post("/v1/runs", json={"input": "hi"})
                run_id = (await resp.json())["run_id"]
                await _wait_for_clarify_event(adapter, run_id)

                # POST with a bogus clarify_id while a real one is pending.
                resp = await cli.post(
                    f"/v1/runs/{run_id}/clarify",
                    json={"clarify_id": "does-not-exist", "answer": "x"},
                )
                assert resp.status == 409
                body = await resp.json()
                assert body["error"]["code"] == "clarify_not_pending"

                # Clean up by resolving the real clarify so the agent thread
                # unwinds and the run completes (otherwise the test fixture
                # teardown can hang for the full timeout).
                from tools import clarify_gateway as _clarify_mod
                _clarify_mod.clear_session(f"run:{run_id}")
                await _wait_for_run_status(cli, run_id, "completed", timeout=3.0)

    @pytest.mark.asyncio
    async def test_missing_clarify_id_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            mock_agent, holder = _make_clarify_agent()
            with patch.object(
                adapter,
                "_create_agent",
                side_effect=_make_create_agent_factory(mock_agent, holder),
            ):
                resp = await cli.post("/v1/runs", json={"input": "hi"})
                run_id = (await resp.json())["run_id"]
                await _wait_for_clarify_event(adapter, run_id)

                resp = await cli.post(
                    f"/v1/runs/{run_id}/clarify",
                    json={"answer": "yes"},
                )
                assert resp.status == 400
                body = await resp.json()
                assert body["error"]["code"] == "missing_clarify_id"

                from tools import clarify_gateway as _clarify_mod
                _clarify_mod.clear_session(f"run:{run_id}")
                await _wait_for_run_status(cli, run_id, "completed", timeout=3.0)

    @pytest.mark.asyncio
    async def test_missing_answer_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            mock_agent, holder = _make_clarify_agent()
            with patch.object(
                adapter,
                "_create_agent",
                side_effect=_make_create_agent_factory(mock_agent, holder),
            ):
                resp = await cli.post("/v1/runs", json={"input": "hi"})
                run_id = (await resp.json())["run_id"]
                evt = await _wait_for_clarify_event(adapter, run_id)

                resp = await cli.post(
                    f"/v1/runs/{run_id}/clarify",
                    json={"clarify_id": evt["clarify_id"]},
                )
                assert resp.status == 400
                body = await resp.json()
                assert body["error"]["code"] == "missing_answer"

                from tools import clarify_gateway as _clarify_mod
                _clarify_mod.clear_session(f"run:{run_id}")
                await _wait_for_run_status(cli, run_id, "completed", timeout=3.0)

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            mock_agent, holder = _make_clarify_agent()
            with patch.object(
                adapter,
                "_create_agent",
                side_effect=_make_create_agent_factory(mock_agent, holder),
            ):
                resp = await cli.post("/v1/runs", json={"input": "hi"})
                run_id = (await resp.json())["run_id"]
                await _wait_for_clarify_event(adapter, run_id)

                resp = await cli.post(
                    f"/v1/runs/{run_id}/clarify",
                    data="not json",
                    headers={"Content-Type": "application/json"},
                )
                assert resp.status == 400

                from tools import clarify_gateway as _clarify_mod
                _clarify_mod.clear_session(f"run:{run_id}")
                await _wait_for_run_status(cli, run_id, "completed", timeout=3.0)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestClarifyAuth:
    @pytest.mark.asyncio
    async def test_requires_auth_when_configured(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs/run_xyz/clarify",
                json={"clarify_id": "abc", "answer": "y"},
            )
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_accepts_valid_bearer_token(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            mock_agent, holder = _make_clarify_agent()
            with patch.object(
                auth_adapter,
                "_create_agent",
                side_effect=_make_create_agent_factory(mock_agent, holder),
            ):
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hi"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                evt = await _wait_for_clarify_event(auth_adapter, run_id)

                resp = await cli.post(
                    f"/v1/runs/{run_id}/clarify",
                    json={"clarify_id": evt["clarify_id"], "answer": "ok"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert resp.status == 200


# ---------------------------------------------------------------------------
# Cleanup: pending clarifies must not survive run completion / cancellation
# ---------------------------------------------------------------------------


class TestClarifyCleanup:
    @pytest.mark.asyncio
    async def test_pending_clarify_is_cleared_on_run_stop(self, adapter):
        """If the run is stopped while a clarify is pending, the blocked
        agent thread must unwind quickly via ``clarify_gateway.clear_session``
        — otherwise the executor pins for the full 10-minute timeout."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            mock_agent, holder = _make_clarify_agent()
            with patch.object(
                adapter,
                "_create_agent",
                side_effect=_make_create_agent_factory(mock_agent, holder),
            ):
                resp = await cli.post("/v1/runs", json={"input": "hi"})
                run_id = (await resp.json())["run_id"]
                await _wait_for_clarify_event(adapter, run_id)

                # Stop the run while clarify is still blocking the agent.
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                # Stop returns 200 or 202 depending on internal state — we
                # care about the *effect*, not the precise status.
                assert stop_resp.status in (200, 202)

                # The run should reach a terminal state (cancelled or
                # completed-after-sentinel) within a couple of seconds.
                # Without the finally-block clear_session call, this
                # would block for ~600s.
                import time as _t
                deadline = _t.monotonic() + 3.0
                while _t.monotonic() < deadline:
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = (await status_resp.json())["status"]
                    if status in {"cancelled", "completed", "stopped", "failed"}:
                        break
                    await asyncio.sleep(0.02)
                else:
                    pytest.fail(
                        f"run {run_id} did not reach a terminal state in 3s "
                        "(clarify cleanup missing?)"
                    )

                # And the clarify session must be gone from the registry.
                assert run_id not in adapter._run_clarify_session_keys
