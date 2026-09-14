#!/usr/bin/env python3
"""Drive OpenHands Cloud over a SusVibes dataset; write predictions.jsonl.

See README.md for the two-adapter shape this harness has to bridge: SusVibes instances are
keyed by a Docker `image_name` (base_commit + task_patch/mask_patch already applied); OpenHands
Cloud has no "run against this arbitrary image" mode. So each instance:

  1. extract_image_to_dir()   -- pull the instance's exact starting tree out of image_name
  2. prepare_repo_branch()    -- push that tree as a scratch branch to `--mirror_repo`
  3. create a bare Cloud conversation (no selected_repository/git_provider -- this deployment
     has no GitHub-App/OAuth linking, only per-secret tokens under Settings -> Secrets), whose
     prompt's first step tells the agent to clone `--mirror_repo` itself using a secret token
     (`--secret_name`, e.g. SUSVIBES_SCRATCH_TOKEN) already present in its sandbox env
  4. wait_for_start_task_ready(), then poll get_conversation()'s `execution_status`
  5. extract_patch()          -- `git diff` inside the live sandbox (no PR needed)
  6. write {instance_id, model_name_or_path, model_patch} to predictions.jsonl

Confirmed end-to-end against openhands.c3ci.cloud with a hand-run smoke test (clone + write +
commit + push all succeeded). Two non-obvious things that cost real debugging time, kept here
as comments so they aren't rediscovered:
  - This deployment's /api/v1/app-conversations rejects the client's default
    `Authorization: Bearer` header with 401 NoCredentialsError; it wants `X-Session-API-Key`.
  - `OpenHandsClient.get_comprehensive_conversation_events()` is unusable here: its app/agent/
    trajectory event-fetching fallbacks 422/500/crash respectively, so it always returns zero
    events. Poll `get_conversation()`'s own `execution_status` field instead.

This is a sketch: the mirror-repo push (step 2) assumes ambient git credentials and does no
retry/locking, and there is no convert.py yet for the standard trajectory format (see README).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import load_dotenv

# By exact path, not the default search -- load_dotenv() with no path walks up from the
# CALLER's file location, which for a real .py file is fine here (this file sits next to
# its own .env), but do it explicitly anyway so it doesn't depend on cwd or import order.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# Both paths must be inserted before either import -- common.py itself needs susvibes'
# repo root on sys.path (it imports susvibes.eval.strategies.prompts at module load time).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import apply_safety_hint, get_instance_template  # noqa: E402
from susvibes.core.utils import resolve_image_name  # noqa: E402

try:
    from remote_openhands import OpenHandsClient, OpenHandsAPIError, OpenHandsTimeoutError
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "evaluation_harness.openhands_cloud requires the `remote_openhands` package "
        "(c3securitytools/aiagent/remote_openhands). Install it first, e.g.:\n"
        "  pip install -e /path/to/c3securitytools/aiagent/remote_openhands\n"
    ) from exc


TERMINAL_ERROR_STATUSES = {"error", "failed", "stuck"}
DEFAULT_TIMEOUT_SEC = 2 * 60 * 60
DEFAULT_POLL_INTERVAL_SEC = 15.0
DEFAULT_SECRET_NAME = "SUSVIBES_SCRATCH_TOKEN"


# ── Client ──


def make_client(timeout: float) -> OpenHandsClient:
    """Build an OpenHandsClient using X-Session-API-Key.

    The client's default `Authorization: Bearer <ACA_API_KEY>` header is rejected by this
    deployment's /api/v1/app-conversations with a 401 NoCredentialsError; it expects the
    same key under X-Session-API-Key instead.
    """
    return OpenHandsClient(
        headers={"X-Session-API-Key": os.getenv("ACA_API_KEY")},
        timeout=timeout,
    )


# ── Step 1: pull the instance's exact starting tree out of the Docker image ──


def extract_image_to_dir(image_name: str, dest: Path, container_work_dir: str = "/project") -> None:
    """Copy `container_work_dir` out of `image_name` into `dest` via a throwaway container."""
    tmp_container = f"ohcloud_extract_{int(time.time())}_{abs(hash(dest))}"
    subprocess.run(
        ["docker", "create", "--pull", "always", "--name", tmp_container, image_name],
        capture_output=True, text=True, check=True,
    )
    try:
        subprocess.run(
            ["docker", "cp", f"{tmp_container}:{container_work_dir}/.", str(dest)],
            check=True,
        )
    finally:
        subprocess.run(["docker", "rm", tmp_container], capture_output=True)


# ── Step 2: push that tree as a scratch branch OpenHands Cloud can clone ──


def prepare_repo_branch(
    instance: dict,
    mirror_repo: str,
    workspace_root: Path,
) -> tuple[str, str]:
    """Extract `instance['image_name']`, commit it, and push it to `mirror_repo`.

    Returns (mirror_repo, branch_name). `branch_name` is derived from `instance_id` and is
    assumed not to already exist on `mirror_repo` (this is a scratch mirror, not the upstream
    project repo) -- pushes with `--force` to make reruns idempotent.
    """
    instance_id = instance["instance_id"]
    branch_name = f"susvibes/{instance_id}"
    local_dir = workspace_root / instance_id
    local_dir.mkdir(parents=True, exist_ok=True)

    image_name = resolve_image_name(instance["image_name"])
    extract_image_to_dir(image_name, local_dir)

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=local_dir, check=True, capture_output=True)

    # Docker images ship a sanitized /project with no .git (see DockerHarness.setup_workspace
    # contract in base.py) -- start a fresh repo at exactly this tree.
    git("init", "-q")
    git("checkout", "-q", "-b", branch_name)
    git("add", "-A")
    git("-c", "user.name=susvibes", "-c", "user.email=susvibes@localhost",
        "commit", "-q", "-m", f"SusVibes starting state for {instance_id}", "--allow-empty")
    git("remote", "add", "origin", f"https://github.com/{mirror_repo}.git")
    git("push", "-q", "-f", "origin", f"HEAD:{branch_name}")

    return mirror_repo, branch_name


def delete_remote_branch(mirror_repo: str, branch_name: str, workspace_root: Path) -> None:
    """Best-effort cleanup of the scratch branch pushed by prepare_repo_branch()."""
    subprocess.run(
        ["git", "push", "-q", f"https://github.com/{mirror_repo}.git", "--delete", branch_name],
        cwd=workspace_root, capture_output=True,
    )


# ── Prompt ──


def build_prompt(instance: dict, mirror_repo: str, branch: str, secret_name: str, strategy: str) -> str:
    repo_name = mirror_repo.split("/")[-1]
    work_dir = f"/workspace/{repo_name}"
    clone_preamble = (
        "Before doing anything else, run exactly these bash commands to prepare your "
        "workspace. If any of them fail, stop and report the error instead of continuing:\n\n"
        f"  git clone https://${{{secret_name}}}@github.com/{mirror_repo}.git {work_dir}\n"
        f"  cd {work_dir}\n"
        f"  git checkout {branch}\n\n"
    )
    problem_statement = instance["problem_statement"]
    if strategy != "none":
        problem_statement = apply_safety_hint(problem_statement)
    return clone_preamble + get_instance_template(work_dir, problem_statement)


# ── Steps 3-5: run the conversation and pull the diff back out ──


def wait_for_completion(
    client: OpenHandsClient,
    conversation_id: str,
    *,
    timeout: float,
    poll_interval: float,
) -> None:
    """Poll get_conversation()'s `execution_status` field until `finished`, or raise.

    Deliberately NOT using get_comprehensive_conversation_events() / events-search: on this
    deployment every fallback that method tries is broken (app: 422 from a `limit=200` the
    server caps at 100; agent: 500; trajectory: a client-side parsing crash), so it silently
    returns zero events forever. get_conversation()'s own execution_status field is unaffected
    by any of that.
    """
    deadline = time.time() + timeout
    latest_status: str | None = None
    while time.time() < deadline:
        latest_status = str(client.get_conversation(conversation_id).get("execution_status") or "").lower()
        if latest_status == "finished":
            return
        if latest_status in TERMINAL_ERROR_STATUSES:
            raise OpenHandsAPIError(
                f"Conversation {conversation_id} ended with execution_status={latest_status}"
            )
        time.sleep(poll_interval)
    raise OpenHandsTimeoutError(
        f"Conversation {conversation_id} did not reach `finished` within {timeout}s "
        f"(last status: {latest_status})"
    )


def extract_patch(client: OpenHandsClient, conversation_id: str, workspace_dir: str) -> str:
    """Run `git diff` inside the live sandbox and return its stdout."""
    result = client.execute_bash_command(
        f"cd {shlex.quote(workspace_dir)} && git diff",
        conversation_id=conversation_id,
    )
    if isinstance(result, dict):
        return (
            result.get("stdout") or result.get("output")
            or result.get("result") or result.get("content") or ""
        )
    return result if isinstance(result, str) else ""


# ── Per-instance orchestration ──


def process_instance(
    instance: dict,
    args: argparse.Namespace,
    workspace_root: Path,
) -> dict[str, Any]:
    instance_id = instance["instance_id"]
    model_name = args.model_name_or_path

    repo, branch = None, None
    client: OpenHandsClient | None = None
    try:
        repo, branch = prepare_repo_branch(instance, args.mirror_repo, workspace_root)

        client = make_client(timeout=args.timeout)
        prompt = build_prompt(instance, repo, branch, args.secret_name, args.strategy)
        client.create_conversation(initial_msg=prompt, agent=args.agent, title=instance_id)
        start_task_id = client.conversation_id

        client.wait_for_start_task_ready(start_task_id, timeout=min(args.timeout, 300))
        cid = client.conversation_id

        wait_for_completion(client, cid, timeout=args.timeout, poll_interval=args.poll_interval)

        repo_name = repo.split("/")[-1]
        model_patch = extract_patch(client, cid, f"/workspace/{repo_name}")

        return {
            "instance_id": instance_id,
            "model_name_or_path": model_name,
            "model_patch": model_patch,
        }
    except Exception as exc:
        return {
            "instance_id": instance_id,
            "model_name_or_path": model_name,
            "model_patch": "",
            "error": str(exc),
        }
    finally:
        if client is not None and client.conversation_id and not args.keep_conversations:
            try:
                client.delete_conversation()
            except Exception:
                pass
            client.close()
        if repo is not None and branch is not None and not args.keep_mirror_branches:
            delete_remote_branch(repo, branch, workspace_root)


# ── CLI ──


def load_instances(dataset_path: Path) -> list[dict]:
    with open(dataset_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_path", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--mirror_repo", type=str, required=True,
        help="owner/name of a scratch GitHub repo the harness can push to.")
    p.add_argument("--secret_name", type=str, default=DEFAULT_SECRET_NAME,
        help="Name of the OpenHands Cloud secret (Settings -> Secrets) holding a GitHub "
             "token scoped to --mirror_repo; the agent uses it to clone.")
    p.add_argument("--agent", type=str, default=None, help="Cloud agent id; default is server-side default.")
    p.add_argument("--model_name_or_path", type=str, default="openhands-cloud")
    p.add_argument("--strategy", type=str, default="none", choices=["none", "generic"])
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SEC)
    p.add_argument("--poll_interval", type=float, default=DEFAULT_POLL_INTERVAL_SEC)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--num_instances", type=int, default=None)
    p.add_argument("--max_workers", type=int, default=4)
    p.add_argument("--keep_conversations", action="store_true",
        help="Don't delete the Cloud conversation/sandbox after each instance (debugging).")
    p.add_argument("--keep_mirror_branches", action="store_true",
        help="Don't delete the pushed scratch branch after each instance (debugging).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    workspace_root = Path(tempfile.mkdtemp(prefix="openhands_cloud_"))

    instances = load_instances(args.dataset_path)
    if args.num_instances is None:
        instances = instances[args.start_idx:]
    else:
        instances = instances[args.start_idx: args.start_idx + args.num_instances]

    # Skip instances already present from a prior (possibly killed) run.
    done_ids: set[str] = set()
    if predictions_path.exists():
        with open(predictions_path) as f:
            done_ids = {json.loads(line)["instance_id"] for line in f if line.strip()}
        instances = [i for i in instances if i["instance_id"] not in done_ids]

    print(f"### PREDICTIONS FILE: {predictions_path} ###")
    print(f"Running {len(instances)} instances ({len(done_ids)} already done)")

    write_lock = Lock()
    n_ok, n_err = 0, 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(process_instance, instance, args, workspace_root): instance["instance_id"]
            for instance in instances
        }
        for future in as_completed(futures):
            instance_id = futures[future]
            result = future.result()
            if result.get("error"):
                n_err += 1
                print(f"  ✗ {instance_id}: {result['error']}")
            else:
                n_ok += 1
                print(f"  ✓ {instance_id}")
            with write_lock:
                with open(predictions_path, "a") as f:
                    f.write(json.dumps(result) + "\n")

    print(f"\nDone: {n_ok} ok, {n_err} errored. Predictions at {predictions_path}")


if __name__ == "__main__":
    main()
