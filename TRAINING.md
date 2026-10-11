# Integrated training

The project's **Annotate / Train** navigation keeps the editor mounted, including
its image, zoom, drafts and undo history. Detection and instance segmentation
have independent review states and datasets. Training continues if the browser
closes. The hosted feature is disabled by default.

## Prepare and train

1. Confirm annotations, then select **Train → Review images**. Mark complete
   images as reviewed individually or together. Reviewed empty images are
   explicit negatives. Editing confirmed objects, their classes or original
   image invalidates the affected review; drafts and proposals are excluded.
2. Preview a split and create a dataset version. The default is 80% training and
   20% validation. **Reserve test set** changes this to 70/20/10. Percentages and
   individual assignments are editable. Identical originals stay together;
   active splits must be nonempty and every annotated class must occur in train.
   Missing validation/test classes are reported. Seed 42 and saved assignments
   make a version reproducible. Changing a split creates another version.
3. Select a compatible model and connected GPU. YOLO26 and RF-DETR offer Nano,
   Small and Medium for both tasks. The initial preference is YOLO26 Small.
   Advanced settings include epochs, batch, resolution, learning rate and
   augmentation. Defaults are 100 epochs and patience 15. Model profiles record
   effective parameters. YOLO mask approximations require explicit acceptance;
   RF-DETR receives COCO RLE with holes and separate components intact.
4. Follow training/validation loss and validation mAP. Mask mAP is primary for
   segmentation; box mAP, precision, recall, per-class AP and events are in
   expandable details. **Pause** saves at an epoch boundary; **Resume** uses the
   original computer, GPU identity, runtime and complete checkpoint. **Cancel**
   terminates the model process. Interrupted executions require explicit resume.

Snapshots retain original bytes and dimensions. Resizing happens inside the
model. The agent's training assignment and image downloads exclude test data;
test never influences early stopping or the selected checkpoint.

Split previews and creation in the editor request compact metadata with
`summary_only=true`; full image annotations stay in the immutable dataset.
Creation sends the preview's `expected_fingerprint` and returns 409 if the
reviewed content or split settings changed. These optional API fields preserve
existing clients and require no persisted schema or agent protocol migration.
Deploy the matching backend before the updated frontend.

## Use results

After the best validation model synchronizes, download its weights and manifest,
try it with an adjustable preview threshold, or preannotate selected images.
Preannotations enter the existing proposal review flow. They retain their model
batch identity and never replace confirmed annotations. Changed source revisions,
dimensions, content or removed classes produce a conflict; request fresh results.

**Evaluate on test** is available only for a completed run with a test partition.
Its saved report identifies the exact snapshot and model SHA-256 and is separate
from the live validation curves. Reopening it returns the saved report. Prediction
and evaluation wait for the run's connected GPU when it is busy.

Final models remain on the connected computer and, in hosted mode, in private
application storage. Failed synchronization can resume from saved outputs without
retraining. Resume checkpoints remain on the original computer. Delete finished
executions and then unused dataset versions to release application storage;
locally retained source copies and agent job directories can be removed when no
remaining run needs them. Disconnecting a computer revokes access, but does not
erase files already downloaded to that computer.

## Optional agent installation

The agent requires Linux NVIDIA x86-64 or ARM64, Python 3.12 and a compatible CUDA
PyTorch runtime. Windows uses WSL2. The editor and CPU hosted image do not install
these dependencies. Use an isolated environment on the user's own computer;
never install training dependencies into a SAM worker environment.

Install the CUDA PyTorch distribution appropriate to the GPU first, then:

```sh
python -m pip install -r requirements-training.txt
python -m app.training.agent provision yolo26-small-detection yolo26-small-segmentation
python -m app.training.agent enroll --server https://annotation.example --name workstation
python -m app.training.agent run
```

Provision only the desired catalog entries; examples for the other family are
`rfdetr-small-detection` and `rfdetr-small-segmentation`. Provisioning is an explicit
download from fixed official URLs with checksum verification. Enrollment prompts
for the one-use code from **Connect computer**; it is valid for ten minutes.
Credentials are stored with mode 0600. The default state directory is
`~/.local/share/annotation-training`; `--state` and `--weights` precede the command.
Keep the directory private and persistent across agent restarts.

For local mode, use `--server http://127.0.0.1:8765` on the same computer running
`run.py`. Local enrollment is associated with the project; separate agent state
directories can be used for separate projects. It requires no account or cloud
service and rejects remote connections. **Release SAM GPU** unloads the local
SAM model before training; manual annotation remains available. Run the local
agent under the same OS user/runtime directory so its GPU lock is shared with SAM.

`Dockerfile.training` accepts `TRAINING_BASE_IMAGE` as an immutable NVIDIA PyTorch
image digest for the target architecture. It adds image libraries and the pinned
optional adapters without changing the CPU server. Mount private persistent state
at `/state`, select the intended GPU, and use `--state /state`. Local containers
also require loopback access to the application and the same host GPU lock
directory; a native isolated environment avoids these container-specific details.
Capture the resulting image digest and installed environment in private release
evidence. Do not put credentials, cached weights or datasets in a Docker context.

Agents use outgoing HTTPS requests only; there are no inbound ports, SSH commands,
bucket credentials or database connections. Each enabled GPU runs one assignment.
Disable unused devices under **Connected computers**. Disabling stops new claims;
disconnecting revokes the credential and interrupts its active work. After a
restore, enroll again using the same state directory to rotate the credential
while preserving the original node identity for checkpoint recovery.

## Runtime and recovery contract

`app/training/runtime-lock.json` pins adapter distributions, wheel hashes and all
12 pretrained weight hashes. Agent enrollment records Python, architecture,
framework versions, adapter/catalog SHA-256 and a hash of installed distributions.
A resumed execution requires the same runtime. Preserve its immutable image or
environment; an upgrade must not silently resume an old optimizer checkpoint.
YOLO runs in full precision to avoid its automatic AMP probe downloading another
model. Telemetry, experiment integrations and automatic package installation are
disabled. Licenses are visible in the catalog: YOLO AGPL/enterprise; these RF-DETR
profiles Apache-2.0. Upstream runtime behavior is isolated in the adapters.

Attempts have 90-second leases and ordered, idempotent updates. The supervisor
stops model processes before a disconnected lease expires; the child also watches
the supervisor. The server fences expired/replaced attempts and never launches a
replacement automatically. A restart resumes only after the user requests it.
Training completion and model synchronization are separate states. Uploads reserve
their full size, accept verified replayable 8 MiB chunks and verify SHA-256 before
publication. The CPU server treats weights as opaque files (maximum 8 GiB).

Detailed framework output stays in each private agent job's `runtime.log`. The
application persists status changes, epochs, metrics and attempts. SSE reconnects
by cursor; periodic reads recover after a stream disconnect. Prediction results
are limited to 20 MiB and proposal mask application to one billion decoded pixels.

## Validation

CPU tests cover local/hosted HTTP flows, reviews, duplicate-safe splits, immutable
versions, RLE conversion, test isolation, upload replay, ownership, quotas,
revision conflicts, watchdogs and synchronization recovery. PostgreSQL tests race
GPU claims across projects and independently available devices. These tests do
not establish model quality or GPU compatibility.

Run the opt-in real-model probe only on authorized disposable GPU environments:

```sh
python -m scripts.validate_training_runtime --output /private-validation/run --weights /private-validation/weights
```

The output must be outside the checkout. It exercises both tasks and families,
with/without test, pauses/resumes real optimization and runs prediction/evaluation.
Use one GPU per invocation; retain reports, runtime/image hashes and logs privately
for both x86-64 and ARM64. Small synthetic probes validate execution, not accuracy
on a user's dataset or memory capacity for every model size. WSL2 follows the
Linux protocol but requires its own installation validation.

For the integrated HTTP/storage/supervisor check, install `fastapi==0.142.2` in
the disposable validation environment (not required by the agent), then run
`python -m scripts.validate_training_cycle --output /private-validation/cycle
--weights /private-validation/weights`. Optional `--family rfdetr --task
segmentation` exercises masks. This test uses real checkpoints and verifies
that applying model proposals preserves confirmed annotations.

Hosted migration and rollback are described in [HOSTED.md](HOSTED.md#training-rollout-and-rollback).
