"""
Sandbox runner: applies a unified diff to the base files, writes generated tests,
then executes both inside an ephemeral Docker container.

Security constraints applied to every test-execution container:
  --network none     no outbound network access
  --memory 256m      capped RAM
  --cpus 0.5         capped CPU
  --read-only        immutable container filesystem (writable /tmp via tmpfs)
  --rm               auto-removed on exit

Test dependencies (pytest, fastapi, httpx) cannot be installed inside that
container — there is no network access and the filesystem is read-only — so
they are baked once into a local image (`_SANDBOX_IMAGE`) via `docker build`,
which is the only step allowed network access. Every actual test run reuses
that cached image fully offline.
"""

import subprocess
import tempfile
from pathlib import Path

from src.agents.tools import BASE_FILES
from src.state import AgentMessage, AgentState, TestResult
from src.utils.compress import strip_ansi, strip_pytest_noise

# Packages pre-installed in the sandbox image. Phase 3: derive from the target
# repo's requirements.txt or pyproject.toml instead of hardcoding.
_SANDBOX_PACKAGES = "pytest fastapi httpx"

_BASE_IMAGE = "python:3.12-slim"
_SANDBOX_IMAGE = "mac-sandbox:latest"

_MAX_RUNTIME_SECONDS = 60
_IMAGE_BUILD_TIMEOUT_SECONDS = 300


def _write_base_files(tmp: Path, base_files: dict[str, str]) -> None:
    for rel_path, content in base_files.items():
        dest = tmp / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)


def _has_unsafe_path(diff: str) -> bool:
    """
    Reject diffs whose file headers could make `patch -p1` write outside `tmp`
    (e.g. via `../` segments or absolute paths in `--- a/...` / `+++ b/...`).
    """
    for line in diff.splitlines():
        if line.startswith(("--- ", "+++ ")):
            target = line[4:].split("\t")[0].strip()
            if target in ("/dev/null",):
                continue
            # Strip the a/ or b/ prefix that `-p1` expects, if present.
            parts = target.split("/", 1)
            rel = parts[1] if len(parts) == 2 and parts[0] in ("a", "b") else target
            if rel.startswith("/") or ".." in Path(rel).parts:
                return True
    return False


def _apply_diff(tmp: Path, diff: str) -> tuple[bool, str]:
    """Run `patch -p1` in tmp. Returns (success, stderr)."""
    if not diff or diff.startswith("(generator"):
        return False, "No diff to apply."

    if _has_unsafe_path(diff):
        return False, "Refused: diff references a path outside the sandbox directory."

    diff_file = tmp / "_patch.diff"
    diff_file.write_text(diff)

    result = subprocess.run(
        ["patch", "-p1", "--input", str(diff_file)],
        cwd=tmp,
        capture_output=True,
        text=True,
    )
    diff_file.unlink(missing_ok=True)
    return result.returncode == 0, result.stderr


def _image_exists(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _ensure_sandbox_image() -> str | None:
    """
    Build the cached sandbox image (network required) if it doesn't exist yet.
    Returns an error message on failure, or None on success.
    """
    if _image_exists(_SANDBOX_IMAGE):
        return None

    dockerfile = f"FROM {_BASE_IMAGE}\nRUN pip install --no-cache-dir {_SANDBOX_PACKAGES}\n"
    try:
        # Build with an empty context (`-`) since the Dockerfile needs no local
        # files — this avoids sending the caller's CWD to the Docker daemon.
        with tempfile.TemporaryDirectory(prefix="mac_build_ctx_") as empty_ctx:
            proc = subprocess.run(
                ["docker", "build", "-t", _SANDBOX_IMAGE, "-f", "-", empty_ctx],
                input=dockerfile,
                capture_output=True,
                text=True,
                timeout=_IMAGE_BUILD_TIMEOUT_SECONDS,
            )
        if proc.returncode != 0:
            return strip_ansi(proc.stderr)[-800:]
        return None
    except subprocess.TimeoutExpired:
        return f"Sandbox image build timed out after {_IMAGE_BUILD_TIMEOUT_SECONDS}s."
    except FileNotFoundError:
        return (
            "Docker not found. Install Docker Desktop (or docker-ce) "
            "and ensure the daemon is running."
        )


def _run_docker(tmp: Path, test_code: str) -> TestResult:
    test_file = tmp / "test_generated.py"
    test_file.write_text(test_code)

    build_err = _ensure_sandbox_image()
    if build_err:
        return TestResult(exit_code=1, stdout="", stderr=f"Sandbox image build failed: {build_err}")

    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--memory", "256m",
        "--cpus", "0.5",
        "--read-only",
        "--tmpfs", "/tmp",
        "-v", f"{tmp}:/workspace:ro",
        "-w", "/workspace",
        _SANDBOX_IMAGE,
        "sh", "-c",
        "pytest test_generated.py -v 2>&1",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_MAX_RUNTIME_SECONDS,
        )
        return TestResult(
            exit_code=proc.returncode,
            stdout=strip_pytest_noise(proc.stdout),
            stderr=strip_ansi(proc.stderr),
        )
    except subprocess.TimeoutExpired:
        return TestResult(
            exit_code=1,
            stdout="",
            stderr=f"Sandbox timed out after {_MAX_RUNTIME_SECONDS}s.",
        )
    except FileNotFoundError:
        return TestResult(
            exit_code=1,
            stdout="",
            stderr=(
                "Docker not found. Install Docker Desktop (or docker-ce) "
                "and ensure the daemon is running."
            ),
        )


def get_patched_files(diff: str, base_files: dict[str, str]) -> tuple[dict[str, str], str]:
    """
    Apply a unified diff to base_files and return the resulting file contents.
    Used by the git agent to build the commit payload without running tests.
    Returns (patched_files_dict, error_message). On failure, error_message is non-empty.
    """
    with tempfile.TemporaryDirectory(prefix="mac_patch_") as tmp_str:
        tmp = Path(tmp_str)
        _write_base_files(tmp, base_files)
        patched, err = _apply_diff(tmp, diff)
        if not patched:
            return {}, err
        result: dict[str, str] = {}
        for f in tmp.rglob("*"):
            if f.is_file() and not f.name.startswith("_"):
                rel = str(f.relative_to(tmp))
                result[rel] = f.read_text(encoding="utf-8", errors="replace")
        return result, ""


def docker_runner_node(state: AgentState) -> dict:
    with tempfile.TemporaryDirectory(prefix="mac_sandbox_") as tmp_str:
        tmp = Path(tmp_str)

        _write_base_files(tmp, BASE_FILES)

        patched, patch_err = _apply_diff(tmp, state["current_diff"] or "")
        if not patched:
            result = TestResult(
                exit_code=1,
                stdout="",
                stderr=f"patch failed: {patch_err}",
            )
        else:
            result = _run_docker(tmp, state["generated_tests"] or "")

    verdict = "PASSED" if result.passed else f"FAILED (exit {result.exit_code})"
    next_status = "approved" if result.passed else "debugging"

    return {
        "test_result": result,
        "status": next_status,
        "messages": [
            AgentMessage(
                role="test_generator",
                content=f"Sandbox {verdict}. "
                        + (result.stdout.splitlines()[-1] if result.stdout else result.stderr[:120]),
            )
        ],
    }
