# Releases and deployment

## Create a candidate

1. Merge the reviewed PR into `main` with all required checks passing.
2. Run **Release candidate** on `main`, supplying an unused `vMAJOR.MINOR.PATCH`
   version (for example `v0.2.0`). The workflow validates the version, reruns CI,
   creates a tag at that exact commit and prepares a **draft** GitHub release.
3. The draft includes tracked source, compiled frontend, a manifest and SHA-256
   checksums. No dataset, weights, environment file or private deployment state
   belongs in an asset. A tag is immutable by convention: fix a bad candidate
   with a new version rather than moving the old tag.

The Git tag is the application release identifier. Docker images additionally
use immutable commit tags/digests. A release does not claim that all components
are currently deployed or that GPU validation has already passed.

## Validate and promote privately

Keep the actual inventory, SSH aliases, deployment scripts, staging/production
records and rollback procedures in the private operations repository. Raw
secrets remain outside that repository too. Public Actions never connect to
personal servers or execute untrusted PRs on self-hosted runners.

For a candidate that changes inference/runtime/protocol behavior, build and
exercise both supported GPU workers with real weights: points, boxes, polygon
seeds, text, Spanish translation and visual references. Verify cancellation,
restarts and fallback when those paths change. Use disposable projects and the
development OAuth/R2 environment, preserving production data and quotas.

Validate upload → annotate → save → reopen → COCO export in local and hosted
modes. Storage/schema changes also require coordinated backup/restore checks.
Retain exact commands, outcomes and limitations in the private release record;
publish a sanitized summary. Documentation/CI-only releases can reference
existing runtime evidence for unchanged application code.

Publish the draft release after its applicable checks pass. Deploy the approved
tag/commit deliberately, never automatically on every push. Record:

- Release tag and source commit; web, dispatcher, maintenance and each worker's
  actual image digest/source revision (unchanged compatible workers can remain).
- Validation evidence, environment, deployment time, backup reference and the
  previous component versions/configuration needed for rollback.
- Schema changes and protocol compatibility, including deployment order.

Worker protocol changes normally roll out to the fallback first, pass readiness
checks, then to the primary and dispatcher in the documented compatible order.
Drain/finish active jobs before replacing a worker. Keep a known-good fallback.

## Recovery

Reverting source is not a database restore. Preserve the prior immutable images
and private launch configuration. Roll back stateless services only when their
protocol/schema compatibility permits it. Recover PostgreSQL, project SQLite
snapshots and the object manifest together when a data restore is required;
follow `HOSTED.md`, never overwrite a live database blindly.

The first hosted release consolidates the previously validated hosted branch.
Future changes use short-lived feature/fix branches from `main` and this same
candidate → validation → publication → deployment process.
