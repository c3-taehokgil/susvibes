"""Mock-based unit tests for the pure logic in run_infer.py.

These do NOT touch a real OpenHands Cloud instance, Docker, or git remotes — they only
exercise wait_for_completion()'s polling, extract_patch()'s response parsing, and
build_prompt()'s template substitution, against a fake OpenHandsClient. Run with:

    pytest evaluation_harness/openhands_cloud/test_run_infer.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_infer  # noqa: E402
from remote_openhands import OpenHandsAPIError, OpenHandsTimeoutError  # noqa: E402


# ── wait_for_completion ──
# Polls client.get_conversation()'s `execution_status` field directly -- NOT
# get_comprehensive_conversation_events(), which is unusable on the real deployment this was
# validated against (see run_infer.py's module docstring).


def test_wait_for_completion_returns_on_finished():
    client = MagicMock()
    client.get_conversation.return_value = {"execution_status": "finished"}
    run_infer.wait_for_completion(client, "cid", timeout=5, poll_interval=0.01)
    client.get_conversation.assert_called_with("cid")


def test_wait_for_completion_polls_until_finished(monkeypatch):
    client = MagicMock()
    client.get_conversation.side_effect = [
        {"execution_status": "running"},
        {"execution_status": "running"},
        {"execution_status": "finished"},
    ]
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    run_infer.wait_for_completion(client, "cid", timeout=5, poll_interval=0.01)

    assert client.get_conversation.call_count == 3
    assert sleeps == [0.01, 0.01]


@pytest.mark.parametrize("bad_status", ["error", "failed", "stuck"])
def test_wait_for_completion_raises_on_terminal_error(bad_status):
    client = MagicMock()
    client.get_conversation.return_value = {"execution_status": bad_status}
    with pytest.raises(OpenHandsAPIError, match=bad_status):
        run_infer.wait_for_completion(client, "cid", timeout=5, poll_interval=0.01)


def test_wait_for_completion_times_out(monkeypatch):
    client = MagicMock()
    client.get_conversation.return_value = {"execution_status": "running"}
    monkeypatch.setattr(time, "sleep", lambda s: None)

    # Force the deadline to already be in the past on the first loop check.
    times = iter([100.0, 100.0, 200.0])
    monkeypatch.setattr(time, "time", lambda: next(times))

    with pytest.raises(OpenHandsTimeoutError):
        run_infer.wait_for_completion(client, "cid", timeout=50, poll_interval=0.01)


def test_wait_for_completion_handles_missing_execution_status_key():
    """A conversation response with no execution_status yet (still provisioning) must not
    crash -- it should be treated as not-yet-finished and keep polling."""
    client = MagicMock()
    client.get_conversation.side_effect = [
        {},  # no execution_status key at all
        {"execution_status": "finished"},
    ]
    run_infer.wait_for_completion(client, "cid", timeout=5, poll_interval=0.01)


# ── extract_patch ──


@pytest.mark.parametrize("key", ["stdout", "output", "result", "content"])
def test_extract_patch_reads_known_dict_keys(key):
    client = MagicMock()
    diff_text = "diff --git a/f.py b/f.py\n+x = 1\n"
    client.execute_bash_command.return_value = {key: diff_text}
    assert run_infer.extract_patch(client, "cid", "/workspace/repo") == diff_text


def test_extract_patch_prefers_stdout_over_other_keys():
    client = MagicMock()
    client.execute_bash_command.return_value = {"stdout": "real diff", "content": "unused"}
    assert run_infer.extract_patch(client, "cid", "/workspace/repo") == "real diff"


def test_extract_patch_handles_plain_string_response():
    client = MagicMock()
    client.execute_bash_command.return_value = "diff --git a/f.py b/f.py\n"
    assert run_infer.extract_patch(client, "cid", "/workspace/repo") == "diff --git a/f.py b/f.py\n"


def test_extract_patch_empty_when_no_changes():
    client = MagicMock()
    client.execute_bash_command.return_value = {"stdout": ""}
    assert run_infer.extract_patch(client, "cid", "/workspace/repo") == ""


def test_extract_patch_quotes_workspace_dir():
    client = MagicMock()
    client.execute_bash_command.return_value = {"stdout": ""}
    run_infer.extract_patch(client, "cid", "/workspace/has space")
    command = client.execute_bash_command.call_args.args[0]
    assert "'/workspace/has space'" in command
    assert command.startswith("cd ")
    assert command.endswith("&& git diff")


# ── build_prompt ──


def test_build_prompt_embeds_problem_statement_and_workspace():
    instance = {"problem_statement": "Fix the SQL injection in query.py"}
    prompt = run_infer.build_prompt(instance, "acme/susvibes-scratch", "susvibes/foo", "MY_TOKEN", strategy="none")
    assert "Fix the SQL injection in query.py" in prompt
    assert "/workspace/susvibes-scratch" in prompt


def test_build_prompt_includes_clone_instructions_with_secret_and_branch():
    instance = {"problem_statement": "Fix the bug."}
    prompt = run_infer.build_prompt(instance, "acme/susvibes-scratch", "susvibes/my-instance", "MY_TOKEN", strategy="none")
    assert "git clone https://${MY_TOKEN}@github.com/acme/susvibes-scratch.git /workspace/susvibes-scratch" in prompt
    assert "git checkout susvibes/my-instance" in prompt


def test_build_prompt_applies_safety_hint_only_for_generic_strategy():
    instance = {"problem_statement": "Fix the bug."}
    none_prompt = run_infer.build_prompt(instance, "acme/repo", "branch", "TOKEN", strategy="none")
    generic_prompt = run_infer.build_prompt(instance, "acme/repo", "branch", "TOKEN", strategy="generic")
    assert generic_prompt != none_prompt
    assert len(generic_prompt) > len(none_prompt)
