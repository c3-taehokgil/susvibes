# OpenHands Cloud Evaluation Harness

Drives **OpenHands Cloud** over a SusVibes dataset and writes `predictions.jsonl`. Validated
end-to-end against `openhands.c3ci.cloud` (a self-hosted OpenHands Cloud deployment); should
also work against the public `app.all-hands.dev` once its own credentials/secret are set up,
though that hasn't been tested here.

Unlike [`evaluation_harness/openhands/`](../openhands) (self-hosted OpenHands 0.54 running in
*your* Docker sandbox from `image_name`), Cloud has no "run against this arbitrary image" mode
at all. This deployment also has **no GitHub App/OAuth linking UI** — there's no "connect your
GitHub account" flow, only per-account secrets under Settings → Secrets. So this harness's job
is: reconstruct each instance's starting tree and push it to a scratch mirror repo, then have
the *agent itself* clone it inside its own sandbox using a secret token — not via any
`selected_repository` API parameter.

```
image_name (Docker)  --extract-->  local worktree  --push branch-->  mirror repo
                                                                          |
                                                                          v
                                    bare OpenHands Cloud conversation (no selected_repository);
                                    prompt's first step: agent clones mirror repo itself using
                                    a secret token already in its sandbox env, checks out the
                                    branch, then works the task -- run to completion, then
                                    `git diff` pulled straight from the live sandbox
                                                                          |
                                                                          v
                                                              predictions.jsonl (model_patch)
```

## Requirements

- The [`remote_openhands`](../../../c3securitytools/aiagent/remote_openhands) client package
  installed (`pip install -e <path-to-c3securitytools>/aiagent/remote_openhands` — or install
  `bou2_openhands` via Poetry, which pulls it in as an editable dependency).
- **Python 3.12** (not the 3.11 the main SusVibes README suggests — `remote_openhands` requires
  `>=3.12`).
- `ACA_API_URL` / `ACA_API_KEY` in a `.env` **in this directory** (next to `run_infer.py` —
  `load_dotenv()` resolves relative to the calling file's location, not your shell's cwd, so
  it has to live here specifically, not e.g. the repo root or a script elsewhere).
- `docker` and `git` on PATH, with local push access to `--mirror_repo`.
- A **scratch mirror repo** (`--mirror_repo owner/name`) — just an empty throwaway repo. It is
  *not* the original upstream project repo.
- A **secret** in OpenHands Cloud Settings → Secrets (name matches `--secret_name`, default
  `SUSVIBES_SCRATCH_TOKEN`) holding a GitHub token scoped to *only* `--mirror_repo`
  (fine-grained PAT, Contents: Read and write is enough). Don't reuse a broader shared
  `GITHUB_TOKEN` secret that's scoped to a different org/project — add a separate one.

## Run

```bash
python run_infer.py \
    --dataset_path ../../datasets/default/susvibes_dataset.jsonl \
    --mirror_repo your-org/susvibes-scratch \
    --secret_name SUSVIBES_SCRATCH_TOKEN \
    --output_dir logs/openhands_cloud/default \
    --max_workers 4
```

Writes `predictions.jsonl` (`{instance_id, model_name_or_path, model_patch}`) under
`--output_dir`, incrementally, so a killed run can resume by filtering out already-completed
`instance_id`s. Feed that file straight into `python -m susvibes.eval.core --predictions_path ...`.

## Two deployment-specific gotchas already worked around here

Found by hand-running a smoke test against `openhands.c3ci.cloud` before trusting the harness
with real instances — kept here so they aren't rediscovered:

1. **Auth header**: this deployment's `/api/v1/app-conversations` rejects the client's default
   `Authorization: Bearer <ACA_API_KEY>` with `401 {"error":"NoCredentialsError"}`. It wants the
   same key under `X-Session-API-Key` instead — see `make_client()`.
2. **Polling completion**: `OpenHandsClient.get_comprehensive_conversation_events()` is
   unusable here — its three internal fallbacks all fail (app events: 422, the library sends
   `limit=200` but the server caps it at 100; agent events: 500; trajectory: a client-side
   parsing bug), so it silently returns zero events forever. `wait_for_completion()` instead
   polls the plain `get_conversation()` call's own `execution_status` field, which isn't
   affected by any of that.

If you're pointing this at a different OpenHands Cloud deployment (e.g. the public
`app.all-hands.dev`), re-verify both of these before trusting a real run — they may behave
differently there (e.g. Bearer auth and/or `selected_repository`/`git_provider` might actually
work on a deployment that does have GitHub App/OAuth linking, in which case the clone-via-secret
approach here is unnecessary complexity you could drop).

## Known gaps (this is a sketch, not a hardened harness)

- `prepare_repo_branch()` pushes with a plain `git push`; it doesn't handle auth, rate limits,
  or repos that already contain the branch name.
- No `convert.py` yet — `predictions.jsonl` is all `susvibes.eval.core` needs, but there's no
  trajectory export to the [standard format](../TRAJECTORY_FORMAT.md) for leaderboard tooling.
- Only validated on a hand-run smoke test (clone + write + commit + push, and a bare
  conversation reaching `execution_status: finished`) — not yet run against a real SusVibes
  instance end-to-end (image extraction → agent fixes a real vuln → patch scored by
  `susvibes.eval.core`).
