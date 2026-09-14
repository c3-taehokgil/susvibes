# OpenHands Cloud Evaluation Harness

Drives **OpenHands Cloud** (the hosted agent product — e.g. `app.all-hands.dev`, or a
self-hosted deployment such as `openhands.c3ci.cloud`) over a SusVibes dataset and writes
`predictions.jsonl`.

Unlike [`evaluation_harness/openhands/`](../openhands) (self-hosted OpenHands 0.54 running in
*your* Docker sandbox from `image_name`), Cloud only runs against a GitHub repo + branch it
already has provider access to — it has no "run against this arbitrary image" mode. So this
harness has one job the other harnesses don't: reconstruct each instance's starting tree
(`base_commit` + `task_patch`/`mask_patch`, i.e. the same state baked into `image_name`) and
push it as a branch to a **scratch mirror repo** before handing it to Cloud.

```
image_name (Docker)  --extract-->  local worktree  --push branch-->  mirror repo
                                                                          |
                                                                          v
                                            OpenHands Cloud conversation (selected_repository
                                            + selected_branch), run to completion, then
                                            `git diff` pulled straight from the live sandbox
                                                                          |
                                                                          v
                                                              predictions.jsonl (model_patch)
```

## Requirements

- The [`remote_openhands`](../../../c3securitytools/aiagent/remote_openhands) client package
  installed (`pip install -e <path-to-c3securitytools>/aiagent/remote_openhands`), plus
  `ACA_API_URL` / `ACA_API_KEY` configured for it (see that package's README).
- `docker` and `git` on PATH.
- Write access to a **scratch mirror repo** that your OpenHands Cloud account/org has GitHub
  provider access to (`--mirror_repo owner/name`). This is *not* the original upstream repo —
  each instance gets pushed there as its own throwaway branch, deleted after the run unless
  `--keep_mirror_branches` is passed. Push auth is whatever `git push` already resolves locally
  (SSH key / credential helper) — this script does not manage credentials itself.

## Run

```bash
python run_infer.py \
    --dataset_path ../../datasets/default/susvibes_dataset.jsonl \
    --mirror_repo your-org/susvibes-scratch \
    --output_dir logs/openhands_cloud/default \
    --max_workers 4
```

Writes `predictions.jsonl` (`{instance_id, model_name_or_path, model_patch}`) under
`--output_dir`, incrementally, so a killed run can resume by filtering out already-completed
`instance_id`s. Feed that file straight into `python -m susvibes.eval.core --predictions_path ...`.

## Known gaps (this is a sketch, not a hardened harness)

- `prepare_repo_branch()` pushes with a plain `git push`; it doesn't handle auth, rate limits,
  or repos that already contain the branch name.
- No `convert.py` yet — `predictions.jsonl` is all `susvibes.eval.core` needs, but there's no
  trajectory export to the [standard format](../TRAJECTORY_FORMAT.md) for leaderboard tooling.
  `client.get_comprehensive_conversation_events()` has everything needed to write one.
- Completion is detected by polling for an `execution_status: finished` event — same signal
  `remote_openhands` uses internally, just read here over the public API instead of its
  private `_wait_until_finished_with_response`.
