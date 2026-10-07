# Hosted deployment

The hosted entry point is independent of `run.py`; local projects keep their
original behavior. The public server contains no PyTorch installation. Google
OAuth, R2 credentials, worker addresses, tokens and GPU UUIDs belong in private
configuration outside the checkout. Never commit a populated environment file.

## CPU server

1. Copy `hosted.env.example` and `postgres.env.example` outside the repository
   and restrict them to the deployment account (`chmod 600`). Set independent
   OAuth clients, bucket-scoped R2 credentials and worker tokens for each
   environment. Set the database URL password to the PostgreSQL password.
2. Configure `ANNOTATION_PRIVATE_ENV_FILE` and `ANNOTATION_POSTGRES_ENV_FILE` as
   absolute filenames in the deployment shell. For production, use HTTPS and
   `ANNOTATION_ENVIRONMENT=prod`, `ANNOTATION_R2_BUCKET=annotation-prod`.
3. Set `ANNOTATION_RELEASE` to the Git commit being deployed, then run:

   ```sh
   docker compose -f deploy/compose.yaml build web
   docker compose -f deploy/compose.yaml up -d
   ```

4. Adapt the Nginx example privately. The application port is bound to localhost;
   PostgreSQL has no published port. Keep Cloudflare R2 public access off and
   bypass shared cache for all authenticated application paths.
5. Check OAuth, private images, inference, export and backup restoration before
   enabling public DNS. Retain the previous image tag for rollback. Restore
   PostgreSQL and project databases together; do not downgrade metadata blindly.

The maintenance container runs the coordinated backup and reconciliation CLI.
All four application processes use the same PostgreSQL metadata and the same
project volume. PostgreSQL and `pg_dump` use major version 15 together.

## Interactive latency

The dispatcher checks the queue and active results every 200 ms, independently
of the five-second retry interval when no GPU is available. Empty queues do not
send GPU health requests. `ANNOTATION_DISPATCHER_POLL_MILLISECONDS` configures
the interactive interval. The browser checks jobs every 500 ms and retains the
one-second retry delay on network errors. Attempt fencing, cancellation, fair
scheduling and the hard execution deadline remain unchanged.

Inference reads originals through the existing 5 GB disposable SHA-256 cache,
shared with authenticated image downloads, avoiding another R2 fetch on each
click. Ownership and active image status are checked before using the cache;
workers still receive validated original pixels without R2 credentials.

Workers retain originals in a 512 MiB RAM LRU for 30 minutes of inactivity
(`ANNOTATION_WORKER_IMAGE_CACHE_BYTES` changes the byte budget). Keys include
project UUID, image ID and verified SHA-256. New attempts send only that identity
and their prompt. A cold/evicted worker returns `424 image_required` before
admitting the attempt; the dispatcher sends the original with the same attempt
ID, deadline and quota reservation. Results and fingerprints remain fenced
across restarts. Pixels never enter the persistent attempt ledger. Roll out the
worker protocol before the CPU dispatcher; workers also accept inline originals
from the previous dispatcher.

Compatible JPEG, PNG and WebP originals are served directly, without recompression
or resizing. Sources with EXIF rotation or formats requiring conversion retain
the original coordinate grid via PNG without rotation metadata. Authorization
and `private, no-store` apply to every image endpoint. The browser additionally
keeps up to four decoded originals (estimated 256 MiB pixel budget) in tab memory
and preloads one next image after the current image finishes. Session/project
changes clear this cache, including pending requests; no persistent browser
storage or public/shared HTTP caching is used.

## Large source images

Project originals have no fixed byte or megapixel ceiling. The account/global
storage quotas still apply. Files over 8 MiB travel in authenticated 8 MiB chunks
so the reverse proxy never needs to accept a full image in one request. Uploads
reserve their entire size before transfer, verify offsets and replayed chunks,
and validate the complete decoded image before committing it to private R2.
Interrupted reservations expire after one hour of inactivity; maintenance clears
their temporary files. No bucket CORS or browser storage credentials are needed.

The editor, SAM 3 and COCO use original dimensions without downsampling the
source. Available memory, the 120-second GPU deadline and annotation output
budgets still apply; removing an image ceiling does not create unlimited RAM.
The separate visual-example prompt keeps its existing input budget.

## GPU workers

The reproducible ARM64 GB10 image is `Dockerfile.worker`, based on NVIDIA
`nvcr.io/nvidia/pytorch:25.10-py3` (Python 3.12, vendor PyTorch 2.9 and torchvision
0.24). It preserves the vendor CUDA framework, installs pinned model dependencies
and checks the full dependency set. The base image is pinned by its verified
registry digest. Both GB10 deployments use this Dockerfile.
Docker hosts need NVIDIA Container Toolkit and a driver compatible with CUDA 13.

Provision the pinned `facebook/sam3` revision and the translation revision from
`app/translation.py` in the model cache using an authorized Hugging Face account.
No model weights or Hugging Face credentials belong in the image or repository.

Install the checkout and virtual environment under `/opt/annotation`, create the
unprivileged `annotation-worker` account, and install `annotation-worker.service`
with its private `/etc/annotation/worker.env`. Restrict the worker port in the
Tailscale policy/firewall to the application server. Bind only to the host's
Tailscale address and select the assigned GPU by UUID. Use the separately
configured fallback host; do not reassign GPUs used by other applications.
Set `ANNOTATION_WORKER_ALLOWED_CLIENTS` to the CPU server's exact Tailscale IP;
the worker checks the socket peer before authentication or reading any body.
The included launcher disables proxy headers so clients cannot forge that peer
address. The Docker worker uses host networking and binds only the exact Tailscale
address plus loopback: Docker bridge/userland proxying must not hide the peer IP.

Use `deploy/compose.worker.yaml` with its private environment filenames, host UID
and GID, model cache and state directory. Ensure the state directory is owned by
that UID before starting; the unprivileged container drops all capabilities.

Run exactly one uvicorn process per worker. The supervisor spawns a warm model
process, performs real point, Spanish text and visual probes, and only then
returns `ready: true` from authenticated `GET /v1/health`. Every operation requires
`Authorization: Bearer <private worker token>`. An example service invocation is:

```sh
python -m app.hosted.worker
```

Model startup may take several minutes. Readiness stays false after failed model
load or probes; the supervisor retries. Inference itself has a hard 120-second
deadline. Timeout or cancellation kills the CUDA child before reporting a
terminal result. A worker ledger prevents duplicate attempts across restarts.
Linux parent-death signaling and systemd control-group cleanup prevent orphaned
model processes.

Worker result payloads expire after 24 hours, with a cleanup sweep every minute;
small attempt identities and fingerprints remain as replay fences. An expired
result cannot trigger another model execution. Each result is capped at 20 MiB,
with a cumulative one-billion-pixel proposal budget; narrower prompts can avoid
that limit. These bounds apply only to the hosted worker.

The dispatcher polls every five seconds, keeps one global job active, rotates
accounts fairly, and attempts at most one infrastructure retry. A network timeout
never starts an immediate second inference: it waits for confirmed termination
or the original deadline plus five seconds. Keep host clocks synchronized.

## Local hosted-mode testing

Run PostgreSQL and set the development private environment with the local Google
client and `annotation-dev`. Build the frontend and run `python run_hosted.py`.
For Vite, keep the backend at port 8765, set the public origin to
`http://localhost:5173`, and use Vite's API proxy. Both registered Google callback
URIs end in `/api/v1/auth/google/callback`. The original `python run.py` requires
none of these hosted credentials.
