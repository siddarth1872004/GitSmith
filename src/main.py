"""
FastAPI entrypoint for the multi-agent code generation pipeline.

Run lifecycle:
  POST /generate              → start a new run (background thread); returns run_id
  GET  /runs/{id}             → poll status (diff, review, test result, PR url)
  POST /runs/{id}/approve     → human approves; git_agent creates branch + PR
  POST /runs/{id}/reject      → human rejects; run terminates without a PR

Observability:
  GET  /runs/{id}/trace       → full span-level trace with token counts and cost
  GET  /stats                 → aggregate metrics across all runs

The graph pauses at interrupt_before=["git_agent"] after tests pass.
MemorySaver persists LangGraph state in-process (server restart clears it).
"""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from src import telemetry
from src.graph import build_graph
from src.state import AgentState

app = FastAPI(title="Multi-Agent Code Generator", version="0.3.0")

_graph = build_graph()
_executor = ThreadPoolExecutor(max_workers=2)
_active: dict[str, asyncio.Future] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _initial_state(run_id: str, feature_request: str) -> AgentState:
    return {
        "run_id": run_id,
        "feature_request": feature_request,
        "plan": None,
        "current_diff": None,
        "review_feedback": None,
        "generated_tests": None,
        "test_result": None,
        "debug_feedback": None,
        "git_branch": None,
        "pr_url": None,
        "messages": [],
        "iteration_count": 0,
        "debug_count": 0,
        "status": "planning",
    }


def _serialize_state(values: dict) -> dict:
    return {
        "status": values.get("status"),
        "diff": values.get("current_diff"),
        "plan": values["plan"].model_dump() if values.get("plan") else None,
        "review": values["review_feedback"].model_dump() if values.get("review_feedback") else None,
        "generated_tests": values.get("generated_tests"),
        "test_result": values["test_result"].model_dump() if values.get("test_result") else None,
        "review_iterations": values.get("iteration_count", 0),
        "debug_iterations": values.get("debug_count", 0),
        "git_branch": values.get("git_branch"),
        "pr_url": values.get("pr_url"),
        "messages": [m.model_dump(mode="json") for m in (values.get("messages") or [])],
    }


def _get_snapshot(run_id: str):
    try:
        snapshot = _graph.get_state(_config(run_id))
    except Exception:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")
    if snapshot is None or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")
    return snapshot


def _run_graph(run_id: str, initial: AgentState) -> None:
    """Executed in a thread-pool worker. Starts the trace and invokes the graph."""
    telemetry.start_run(run_id, initial["feature_request"])
    try:
        result = _graph.invoke(initial, config=_config(run_id))
        final_status = result.get("status", "unknown")
        pr_url = result.get("pr_url")
    except Exception as exc:
        final_status = "failed"
        pr_url = None
        raise exc
    finally:
        # Only mark finished if the graph actually terminated (not just interrupted).
        snapshot = _graph.get_state(_config(run_id))
        if not (snapshot and snapshot.next):
            telemetry.finish_run(run_id, final_status, pr_url)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    feature_request: str


@app.post("/generate", status_code=202)
async def start_generate(request: GenerateRequest) -> dict:
    """Start a new generation run. Returns immediately; poll GET /runs/{run_id}."""
    run_id = str(uuid.uuid4())
    initial = _initial_state(run_id, request.feature_request)

    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(_executor, _run_graph, run_id, initial)
    _active[run_id] = future
    future.add_done_callback(lambda _: _active.pop(run_id, None))

    return {"run_id": run_id}


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    """
    Poll the current run state.

    Watch for status == "awaiting_approval": that means tests passed and a
    human needs to call POST /runs/{run_id}/approve or /reject.
    """
    snapshot = _get_snapshot(run_id)
    result = _serialize_state(snapshot.values)

    if snapshot.next and "git_agent" in snapshot.next:
        result["status"] = "awaiting_approval"

    result["running"] = run_id in _active
    return result


@app.post("/runs/{run_id}/approve")
async def approve_run(run_id: str) -> dict:
    """Human approves — resumes graph so git_agent creates the branch and PR."""
    snapshot = _get_snapshot(run_id)
    if "git_agent" not in (snapshot.next or []):
        raise HTTPException(status_code=409, detail="Run is not awaiting approval.")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: _graph.invoke(None, config=_config(run_id)),
    )

    telemetry.finish_run(run_id, result.get("status", "pr_created"), result.get("pr_url"))
    return _serialize_state(result)


@app.post("/runs/{run_id}/reject")
async def reject_run(run_id: str) -> dict:
    """Human rejects — terminates run without creating a PR."""
    snapshot = _get_snapshot(run_id)
    if "git_agent" not in (snapshot.next or []):
        raise HTTPException(status_code=409, detail="Run is not awaiting approval.")

    _graph.update_state(_config(run_id), {"status": "rejected"})

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: _graph.invoke(None, config=_config(run_id)),
    )

    telemetry.finish_run(run_id, "rejected")
    return _serialize_state(result)


@app.get("/runs/{run_id}/trace")
async def get_trace(run_id: str) -> dict:
    """Full span-level trace: per-node timing, token counts, and estimated cost."""
    trace = telemetry.get_trace(run_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"No trace found for run {run_id!r}.")
    return trace.to_dict()


@app.get("/stats")
async def get_stats() -> dict:
    """Aggregate metrics across all runs in this server session."""
    return telemetry.aggregate_stats()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "active_runs": len(_active)}


@app.get("/runs", include_in_schema=False)
async def list_runs() -> dict:
    """Return all runs with summary data for the history page."""
    traces = telemetry.get_all_traces()
    return {
        "runs": [
            {
                "run_id": t.run_id,
                "feature_request": t.feature_request,
                "final_status": t.final_status,
                "started_at": t.started_at.isoformat(),
                "ended_at": t.ended_at.isoformat() if t.ended_at else None,
                "duration_seconds": t.duration_seconds,
                "total_input_tokens": t.total_input_tokens,
                "total_output_tokens": t.total_output_tokens,
                "total_cost_usd": round(t.total_cost_usd, 6),
                "pr_url": t.pr_url,
            }
            for t in sorted(traces, key=lambda x: x.started_at, reverse=True)
        ]
    }


_UI_PATH = Path(__file__).parent.parent / "ui" / "index.html"


@app.get("/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
def ui_root():
    """Serve the single-page UI."""
    if _UI_PATH.exists():
        return FileResponse(_UI_PATH)
    return HTMLResponse("<h1>UI not found</h1><p>Make sure ui/index.html exists.</p>", status_code=404)
