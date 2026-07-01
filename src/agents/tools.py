"""
Tool implementations used by the generator's ReAct loop.

Modes (selected automatically based on environment variables):

  Stub mode  — default, no env vars required.
              read_file reads from BASE_FILES; search_codebase returns a placeholder.

  RAG mode   — activated when TARGET_REPO_PATH is set.
              read_file reads from disk; search_codebase queries a Chroma collection.
              Requires the index to have been built first via scripts/index_repo.py.

All file content is compressed before being returned (comments stripped,
blank lines squeezed, per-result token budget enforced) to minimise context usage.
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb

from src.utils.compress import markitdown_file, strip_python, token_budget

# ---------------------------------------------------------------------------
# Stub file registry — used in stub mode AND by the sandbox runner.
# ---------------------------------------------------------------------------

BASE_FILES: dict[str, str] = {
    "src/__init__.py": "",
    "src/main.py": (
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n"
        "@app.get('/')\n"
        "def root():\n"
        "    return {'status': 'ok'}\n"
    ),
    "src/models.py": (
        "from pydantic import BaseModel\n\n"
        "class Item(BaseModel):\n"
        "    id: int\n"
        "    name: str\n"
    ),
}

_FILE_TOKEN_BUDGET = 700   # per read_file call
_BINARY_TOKEN_BUDGET = 500  # for markitdown-converted docs

# ---------------------------------------------------------------------------
# Lazy RAG initialisation
# ---------------------------------------------------------------------------

_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection | None:
    global _collection
    if _collection is not None:
        return _collection

    repo_path = os.environ.get("TARGET_REPO_PATH")
    if not repo_path:
        return None

    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", ".chroma")
    collection_name = Path(repo_path).resolve().name

    try:
        from src.rag.indexer import load_collection
        _collection = load_collection(persist_dir=persist_dir, collection_name=collection_name)
        return _collection
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _compress_file(content: str, path: str) -> str:
    """Apply appropriate compression based on file extension."""
    if path.endswith('.py'):
        return token_budget(strip_python(content), _FILE_TOKEN_BUDGET)
    return token_budget(content, _FILE_TOKEN_BUDGET)


def read_file(path: str) -> str:
    repo_path = os.environ.get("TARGET_REPO_PATH")
    if repo_path:
        repo_root = Path(repo_path).resolve()
        full = (repo_root / path).resolve()
        try:
            full.relative_to(repo_root)
        except ValueError:
            return f"# {path}: outside of {repo_path} (refused)"
        if not full.is_file():
            return f"# {path}: not found in {repo_path}"

        # Try markitdown for binary/HTML docs first
        converted = markitdown_file(full)
        if converted is not None:
            return token_budget(converted, _BINARY_TOKEN_BUDGET)

        try:
            content = full.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"# Error reading {path}: {e}"

        return _compress_file(content, path)

    # Stub mode
    if path in BASE_FILES:
        return _compress_file(BASE_FILES[path], path)
    return f"# {path}: not in stub registry (set TARGET_REPO_PATH for real file I/O)"


def search_codebase(query: str) -> str:
    collection = _get_collection()
    if collection is not None:
        from src.rag.retriever import search
        return search(collection, query)

    return (
        f"# Stub search for: {query}\n"
        "# Set TARGET_REPO_PATH and run scripts/index_repo.py to enable real search.\n"
        "# No results returned."
    )


TOOL_DISPATCH: dict[str, object] = {
    "read_file": read_file,
    "search_codebase": search_codebase,
}
