# GitSmith-AI: Multi-Agent Autonomous PR Generation Pipeline


A multi-agent code generation pipeline that takes a plain-English feature request and produces a reviewed, tested, and committed pull request — fully automated with a human approval gate before any code is pushed.

Built with LangGraph, FastAPI, and a unified LLM client that supports Anthropic, OpenAI, Gemini, Grok, OpenRouter, and Ollama — swap providers with a single env var.

---

## How it works

```
feature request
      |
   Planner          Reads the request, outputs a structured plan (files, functions, constraints)
      |
  Generator         Explores the codebase via tools (read_file, search_codebase), proposes a unified diff
      |
   Reviewer         Reviews the diff against the plan, classifies issues (bug / security / style)
      |
  [loop if rejected, up to 2x]
      |
 Test Generator     Writes pytest tests for the implemented feature
      |
 Docker Runner      Executes tests in an isolated container
      |
  [loop via Debugger if tests fail, up to 1x]
      |
 --- PAUSE: human approves or rejects ---
      |
  Git Agent         Creates a branch, commits the diff, opens a GitHub pull request
```

Each agent is a LangGraph node. The graph pauses before `git_agent` so a human can review the diff and test results before any code is pushed.

---

## Providers

Set `LLM_PROVIDER` in `.env` to choose which API to use. Only the matching key is required.

| Provider | `LLM_PROVIDER` | Key variable | Default model |
|---|---|---|---|
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | claude-sonnet-4-6 |
| OpenAI | `openai` | `OPENAI_API_KEY` | gpt-4o |
| Google Gemini | `gemini` | `GEMINI_API_KEY` | gemini-2.0-flash |
| xAI Grok | `grok` | `GROK_API_KEY` | grok-3-mini |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | anthropic/claude-sonnet-4-6 |
| Ollama (local) | `ollama` | _(none)_ | llama3.1 |

Override the model with `LLM_MODEL=<model-id>`.

---

## Setup

**Prerequisites:** Python 3.11+, Docker (for sandbox test execution), Git

```bash
git clone https://github.com/siddarth1872004/agentforge
cd agentforge

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[rag,mcp]"
```

**Configure `.env`:**

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Optional: GitHub PR creation
GITHUB_TOKEN=ghp_...
GITHUB_REPO=your-username/your-repo

# Optional: RAG (semantic search over your target codebase)
TARGET_REPO_PATH=/path/to/repo
CHROMA_PERSIST_DIR=/path/to/repo/.chroma
```

**Optional — index a codebase for RAG:**

```bash
python scripts/index_repo.py /path/to/your/target/repo
```

This chunks and embeds the target repo into a local Chroma vector store. The generator will use semantic search to find relevant code before writing diffs.

---

## Running

```bash
source .venv/bin/activate
PYTHONPATH=. uvicorn src.main:app --port 8000 --reload
```

Open **http://localhost:8000** for the UI, or use the REST API directly.

---

## UI

The single-page app at `/` provides:

- **Dashboard** — stat cards (total runs, success rate, active, PRs) and a recent runs table
- **New Run** — text input for the feature request with example prompts
- **Run Monitor** — live pipeline visualization with node status (pending / running / done), tabbed views for diff (side-by-side), test code + sandbox output, and per-agent timing/token charts
- **Approve / Reject** — banner appears when tests pass and the pipeline is paused; approve to open the PR, reject to discard
- **History** — all runs with status, cost, and PR links
- **Stats** — aggregate token usage, cost, and outcome charts

The sidebar footer shows the active provider and model, updated live from `/stats`.

---

## REST API

| Method | Path | Description |
|---|---|---|
| `POST` | `/generate` | Start a run. Body: `{"feature_request": "..."}`. Returns `{"run_id": "..."}`. |
| `GET` | `/runs/{id}` | Poll status. Returns full state including diff, plan, review, test result. |
| `POST` | `/runs/{id}/approve` | Resume the paused graph; `git_agent` creates the branch and PR. |
| `POST` | `/runs/{id}/reject` | Terminate without creating a PR. |
| `GET` | `/runs/{id}/trace` | Full span-level trace: per-agent timing, token counts, and cost. |
| `GET` | `/runs` | List all runs with summary data. |
| `GET` | `/stats` | Aggregate metrics across all runs in the current session. |
| `GET` | `/health` | Liveness check. Returns `{"status": "ok", "active_runs": N}`. |

**Example:**

```bash
# Start a run
RUN=$(curl -s -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"feature_request": "Add a /ping endpoint that returns {\"pong\": true}"}' \
  | jq -r .run_id)

# Poll until awaiting_approval
watch -n2 "curl -s http://localhost:8000/runs/$RUN | jq .status"

# Approve
curl -s -X POST http://localhost:8000/runs/$RUN/approve | jq .pr_url
```

---

## MCP Server

The pipeline is also exposed as an MCP tool server, making it usable from Claude Desktop or any MCP client:

```bash
PIPELINE_API_URL=http://localhost:8000 python -m src.mcp_server
```

Tools exposed: `generate_code`, `get_run_status`, `approve_run`, `reject_run`, `get_run_trace`, `get_stats`.

---

## Project structure

```
src/
  agents/
    _client.py          Unified LLM client (Anthropic + OpenAI-compatible providers)
    planner.py          Structured plan from feature request
    generator.py        ReAct loop: read files, search codebase, propose diff
    reviewer.py         Code review with issue severity classification
    test_generator.py   Pytest file generation
    debugger.py         Root cause analysis on test failures
    tools.py            Tool implementations (read_file, search_codebase)
    git_agent.py        Branch creation and GitHub PR
  rag/
    chunker.py          File chunking for indexing
    indexer.py          Chroma vector store management
    retriever.py        Semantic search over indexed codebase
  sandbox/
    runner.py           Docker-based test execution
  utils/
    compress.py         Context compression (Python stripping, diff reduction, markitdown)
  graph.py              LangGraph state machine with conditional edges
  main.py               FastAPI app — REST API + UI serving
  mcp_server.py         FastMCP server wrapping the REST API
  state.py              Pydantic models and AgentState TypedDict
  telemetry.py          Per-node timing, token counting, cost tracking

ui/
  index.html            Single-page app (Chart.js + diff2html, no build step)

scripts/
  index_repo.py         Index a codebase into Chroma for RAG
```

---

## Cost reference

Approximate cost per full pipeline run (planner → PR) with no retries:

| Provider / Model | Estimated cost |
|---|---|
| Gemini 2.0 Flash | ~$0.002 |
| GPT-4o Mini | ~$0.005 |
| GPT-4o | ~$0.08 |
| Claude Sonnet 4.6 | ~$0.10 |
| Grok 3 Mini | ~$0.01 |
| Ollama (local) | $0.00 |

Actual cost varies with codebase size, number of tool calls, and retry iterations. The Trace tab shows exact per-agent token usage and cost after each run.

---

## License

MIT
