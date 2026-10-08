# Development workflow

Annotation has one `main` branch and two runtime modes. The shared editor,
geometry, masks and COCO behavior should evolve together. Runtime adapters own
local filesystem access versus hosted authentication, object storage and jobs.

## Make a change

1. Check `git status`, fetch `origin`, then create `codex/<feature-or-fix>` from
   `origin/main`. Continue related work on that branch; use a new branch for an
   unrelated task. A stacked PR must name its temporary base.
2. Make the change and a focused regression test. Use synthetic or disposable
   data. Preserve both execution modes.
3. Run the applicable commands in `AGENTS.md` and open a PR. Describe which modes
   changed and any schema/protocol compatibility implications.
4. Merge after the required `CI` checks pass and review is complete. Temporary
   branches may be deleted after merge; published release tags are retained.

`AGENTS.md` supplies instructions to coding agents; GitHub branch protection
enforces the PR/check requirements for everyone. Neither substitutes for review.

## Reproduce CI

Use Python 3.12 and Node 22.13+:

```sh
python -m pip install -r requirements-dev.txt -r requirements-hosted.txt
python -m pytest -q
npm --prefix frontend ci
npm --prefix frontend test
npm --prefix frontend run build
node --experimental-strip-types --test tests/frontend-masks.mjs
```

The Python contracts use CPU tensors and test doubles, without model downloads,
OAuth accounts or real R2 credentials. On Linux CI, install the pinned torch and
torchvision versions from the official CPU wheel index first. CUDA deployment
continues to use its verified vendor runtime and `requirements-worker.txt`.

The PostgreSQL job runs separately against an ephemeral PostgreSQL 15 service.
`ANNOTATION_TEST_DATABASE_URL` enables concurrency checks;
`ANNOTATION_TEST_POSTGRES_URL_FILE` points to a private file holding an admin
DSN for the backup/restore check. This test creates and drops disposable
databases. Never configure either variable with a production connection.

The six PostgreSQL checks are expected skips when those explicit variables are
absent locally. In CI they must all execute. Real GPU/model validation remains a
separate release check, documented in `VALIDATION.md` and the private runbook.

CI also builds the hosted Docker image and verifies that it imports without
PyTorch. Full-history Gitleaks scanning runs on public CI; review additional
infrastructure identifiers manually because secret scanners do not detect them
all. Public PR jobs have read-only tokens and no deployment credentials.

## Configuration and releases

Commit only `.env.example` templates. Keep populated configuration and sensitive
operational evidence outside the checkout. `.gitignore` and `.dockerignore` are
additional safeguards, not secret storage.

See [RELEASING.md](RELEASING.md) for draft releases, private staging validation,
component version records and rollback. Merging a PR does not deploy production.
