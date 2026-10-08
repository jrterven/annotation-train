# Working on Annotation

## Branches and scope

- Keep one codebase for local and hosted modes. Use a short-lived `codex/<change>`
  branch for each feature or fix, based on current `origin/main`. Reuse the branch
  when continuing that same change. Explicitly document a different base for a
  stacked PR. Never commit directly to `main` or keep separate permanent local
  and hosted branches.
- Check the working tree before switching; preserve unrelated user changes.
  Integrate through a PR with passing required checks. Merge, release and deploy
  within the user's authorized scope; an instruction to edit code alone does not
  authorize production deployment. Do not request approval again for an action
  the user has already authorized.
- Explain the user-visible behavior, affected modes, validation and remaining
  limitations in the PR. Follow `CONTRIBUTING.md` and `RELEASING.md`.

## Shared behavior and boundaries

- Reuse editor, geometry, RLE masks, COCO and annotation persistence. Preserve the
  original pixel grid and annotation revisions. `run.py` must work without cloud
  accounts; `run_hosted.py` must remain CPU-only.
- Hosted access uses ownership checks, authenticated downloads, CSRF and private
  storage. Workers accept validated image bytes/identities, never arbitrary
  server paths or URLs. Preserve durable jobs, attempt fencing and quota rules.
- Read `HOSTED.md` and `deploy/README.md` before changing worker protocols, queues,
  storage or deployment. Describe migration and rollback implications for any
  persisted schema or protocol change; support a rolling deployment when needed.

## Validation

- Python changes: `python -m pytest -q` in the project Python 3.12 environment.
- Frontend changes: `npm --prefix frontend test` and
  `npm --prefix frontend run build`. Mask changes also need
  `node --experimental-strip-types --test tests/frontend-masks.mjs`.
- CI runs the local/hosted Python contracts, frontend suite, six real PostgreSQL
  concurrency/restore checks, a CPU image smoke check and a full-history secret
  scan. Do not hide failures or treat skipped GPU validation as a successful GPU
  run. Add a focused regression for behavioral fixes.
- Real-model checks run separately on explicitly configured test GPU workers,
  using disposable projects. Never point test DSNs at production. When changing
  inference/runtime/protocol code, validate supported primary and fallback
  architectures with real weights before production and record private evidence.
- Documentation-only changes need link/command review and relevant workflow
  validation, not a new model benchmark.

## Public repository and private operations

- This repository and all its branches, CI logs and release assets are public.
  Commit only generic templates. Never commit populated environment files,
  credentials, private hosts/addresses, GPU identifiers, account inventories,
  user datasets, databases, model weights or operational logs.
- Keep actual inventory, deployment manifests and runbooks in the separate
  private operations repository. Keep raw secrets outside **both** repositories.
  Do not copy temporary operational folders wholesale into Git.
- Public CI uses GitHub-hosted runners and disposable credentials. It must never
  execute PR code on personal GPU/server runners or receive production secrets.
- Releases identify immutable source commits. Production deployment is a
  separate operation with recorded component versions, checks and rollback;
  never deploy a moving branch or `latest` tag.

## Code Review Rules

- Flag cloud dependencies introduced into the local path, duplicated editor
  logic, lost full-resolution masks or weakened revision checks on writes.
- Flag ownership/CSRF bypasses, public image caching, unfenced retries, quota
  duplication, or secrets/infrastructure details reaching public artifacts.
- A green CPU suite proves contracts, not SAM quality or CUDA compatibility.
