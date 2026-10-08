# Local validation · October 2, 2026

Hardware: MacBook Pro M4 Max, 128 GB, macOS 27.0.1. Isolated Python 3.12.14 environment.

## Completed checks

- `python -m pytest -q`: **194 passing tests** covering persistence, geometry, COCO, API, SAM 3 contracts, polygon drafts, mask-seeded refinement, translation, hole cleanup, and visual-reference uploads and inference contracts.
- `npm run test`: **44 passing tests** covering masks, autosave, conflicts, stale responses, polygon workflows, English interface controls, hole cleanup, real React Konva rendering, hand-tool/Space panning, and visual-reference selection and searches.
- `npm run build`: TypeScript compilation and the Vite production build pass.
- `node --experimental-strip-types --test tests/frontend-masks.mjs`: **6 passing tests**. Fixtures encoded by pycocotools are compared pixel by pixel with the browser decoder.
- `pip check`: no incompatible dependencies reported in the current installed environment.
- A clean resolution of `requirements.txt` using `pip --dry-run --ignore-installed --only-binary=:all:` passed for macOS ARM/Python 3.12; report: `output/qa/install-resolution.json`. This check preceded the addition of translation dependencies and does not establish a fresh installation of the updated requirements. Dependencies do not use local filesystem paths.
- PyTorch 2.14.1 and torchvision 0.29.1 import successfully; Transformers 5.18.0 exposes `Sam3Model` and `Sam3TrackerModel`.
- Metal/MPS is available outside the sandbox; a real `torchvision.ops.roi_align` check on MPS tensors passed.
- Pinned SAM checkpoint: `facebook/sam3`, revision `3c879f39826c281e95690f02c7821c4de09afae7`.

Tests cover rectangular masks, holes, islands, single-pixel objects, image borders, manual geometry and vertex edits, optimistic revisions, concurrent saves, COCO IDs, moving projects, external image roots, relinking, and isolation between browser tabs.

Chrome checks include COCO import, image navigation, vertex dragging/insertion/deletion, canvas panning over controls, undo/redo, reload, and export. Undo restored the exact original mask; redo and reload preserved the edit. Both exported COCO files reconstructed pixel-identical masks using `pycocotools.COCO.annToMask`, including the hole. These checks used synthetic test images, not simulated SAM predictions.

The **Pan image** hand tool was checked in Chrome at 1024 and 1440 pixels wide, beside **Select and edit**. With a selected annotation and 195% zoom, the image followed a 90 × −50 pixel drag; the cursor changed from grab to grabbing and back. Tooltips and the Space shortcut worked, and the saved image state remained byte-identical. Automated Konva tests verify that panning does not trigger annotation callbacks and that returning to an annotation tool preserves original-pixel coordinates. Screenshots: `output/playwright/pan-tool.png` and `pan-tool-1024.png`.

A draft with positive/negative points and a second boxed part also survived navigation and project recovery while inference failed because model access was unavailable. That case validates draft persistence, not segmentation quality or model output.

## Real inference: passed on MPS and CPU

Access to the weights was approved and the official checkpoint of 3,439,938,512 bytes was downloaded. The initial test blocked by HTTP 403 remains recorded in `output/qa/sam3-benchmark.json`; subsequent tests use the real model and resolve that access blocker.

Image: `test_image.jpg`, 1280 × 720, from [Meta's official example](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb), with the concept `shoe`. The model found **12 shoes**. **13 cases on MPS and 13 on CPU** completed: text, positive and negative points, independent boxes, union of two parts, text → clicks, refinement from RLE without caches, and repeated requests using cached image features. MPS remained on Metal throughout its cases, without CPU fallback.

CPU and MPS masks were pixel-identical for every case on this image. Visual inspection confirmed that the proposals correspond to shoes; no reference masks were available to measure boundary accuracy.

| Engine measurement | MPS | CPU |
| --- | ---: | ---: |
| Load from local weights | 8.39 s | 3.91 s |
| First text request | 6.24 s | 5.23 s |
| First click request | 4.43 s | 4.20 s |
| Text with cached image, median | 152 ms | 659 ms |
| Clicks with cached image, median | 18 ms | 28 ms |

Medians use three repetitions. Engine timings include device synchronization but exclude polygon conversion, network transport, and interface rendering. The largest sampled MPS driver memory value was **6.75 GB**, with 5.47 GB of allocated tensors. Measurements were taken after each case; they are not a continuously measured peak or a prolonged memory test.

The negative click changed the mask but did not exclude the exact selected pixel near the boundary in this example. The negative label and coordinates were verified at model input: prompts guide predictions rather than enforcing hard pixel constraints. Demanding boundaries still require review and, where appropriate, manual editing. Refinement from a recovered RLE differed by 16 pixels from refinement using the original logits; reopening the project without rerunning inference preserves the exact mask.

In Chrome, 11 proposals were accepted together; the remaining proposal was refined with positive/negative clicks and confirmed with Enter. The other 11 objects remained unchanged. All 12 objects were saved, and COCO export reconstructed their masks exactly with `pycocotools`. Restarting the server and reopening the project recovered the same complete state.

Two validations were corrected during review: descriptions exceeding the encoder's 32-token limit and non-finite presence values. Seven additional regression tests pass. The updated engine was checked with real weights: it returns HTTP 422 with a clear instruction to shorten a long prompt and still produces 12 proposals for `shoe`, without changing saved annotations.

Local evidence: `output/qa/sam3-real/mps.json`, `cpu.json`, `cpu-mps-comparison.json`, `web-verification.json`, and the `mps-masks`/`cpu-masks` folders containing PNG and RLE files. Demonstration project: `output/qa/sam3-web`; its images are outside the project in `output/qa/sam3-real`.

To repeat with your own data:

```sh
.venv/bin/python scripts/benchmark_sam3.py --image /path/to/image.jpg --text tomato --device mps --output output/sam3-mps.json --artifacts-dir output/sam3-mps-masks
```

## Polygon → Refine with SAM

Tested in Chrome with the official model on MPS: an eight-vertex polygon around a person, rasterization as an initial mask, refinement without clicks or a box, correction with one negative and one positive click, Enter to confirm, and a subsequent vertex edit while zoomed in. The initial mask contained 42,906 pixels; SAM produced 25,697 pixels and the corrections produced 24,078. These are areas from this sample, not accuracy measurements.

A two-vertex draft was recovered after navigation and reload. Confirmation preserved the exact SAM mask; generating editable controls did not change it. Manual editing changed the mask, and COCO reconstruction using `pycocotools.COCO.annToMask` matched the saved mask pixel by pixel. Undo restored the exact previous mask.

Automated regressions cover rejection of self-intersecting geometry without losing the drawing, resuming with N and the continue button, late responses after undo and navigation, coordinates under zoom, and repeated drags against image borders. The welcome illustration and promotional welcome/empty-state copy have been removed.

Local evidence: project `output/qa/polygon-ui`, states `output/qa/polygon-ui-*.json`, and screenshots `output/playwright/polygon-*.png`.

## English interface and local translation

The interface now uses English labels, messages, and tooltips. The text prompt is in the toolbar, outside the image canvas, with an English/Spanish language selector. Chrome checks passed for tooltips activated by both mouse hover and keyboard focus. Mask overlays now show boundaries for all annotations, drafts, and proposals, including holes and disconnected components. Automated Konva checks verify opacity, mask visibility, and constant screen-width outlines under zoom.

The public `Helsinki-NLP/opus-mt-es-en` translation model was downloaded, approximately **312 MB**, at fixed revision `c96e2c5399ebfae4fc43d9669556b9afa74bb69d`. It runs locally on CPU. After the download, the first model load and translation of `zanahorias` → `carrots` took **4.162 s**. Five subsequent short prompts took **51–79 ms**. Recorded examples include `tomates maduros` → `ripe tomatoes`, `hojas verdes` → `green leaves`, `células` → `cells`, `perros y gatos` → `dogs and cats`, and `zapatos` → `shoes`. These examples demonstrate functioning translation on this sample, not general translation-quality metrics.

A real MPS browser run used `sample/images/1001.jpg` in the separate QA project `output/qa/international-ui`. A Spanish prompt (`zanahorias`, translated to `carrots`) and an English prompt (`carrots`) each produced **19 proposals with pixel-identical masks**. One proposal was accepted and exported; reconstruction from COCO matched its saved mask exactly. The English prompt used by SAM was visible in the interface for review. Proposal counts and equality establish workflow consistency, not segmentation accuracy against ground truth.

Chrome layout checks passed at viewport widths of 1024, 1440, and 1920 pixels. The prompt remains outside the canvas and the toolbar adapts without horizontal page overflow. Hover tooltips disappear after mouse clicks and reappear on keyboard focus.

The original `sample` project was reopened without supplying an image root. Its project metadata and all seven saved image states were compared with the pre-test baseline and remained unchanged.

Local evidence: translation results in `output/qa/translation-validation.json`, workflow checks in `output/qa/international-ui-verification.json`, the project `output/qa/international-ui`, and screenshots `output/playwright/english-*.png`.

## Small-hole cleanup

Read-only inspection of the lower carrot in `sample/images/1001.jpg` found four actual mask holes of 1, 1, 2, and 4 pixels. Each hole had four editable vertices, producing 16 interior handles. These controls describe excluded mask pixels; polygon conversion did not invent the holes.

The explicit **Fill small holes** action uses an adjustable area threshold in original-image pixels (16 px² by default). Tests verify that larger holes, exterior background, and existing foreground pixels remain unchanged, including disconnected parts and islands inside holes. Background uses 4-connectivity. The endpoint computes a result without saving it; the frontend applies it as an undoable, autosaved edit to the selected instance. Tests also cover invalid limits, no-op/failure history, changed classes, late responses, and a hole enclosed by separate foreground components whose contours have no interior rings.

Chrome validation used a separate QA project referencing the original image. Cleanup filled exactly eight pixels, removed all four tiny holes and their interior handles, and retained every foreground pixel. The exported COCO reconstructed the cleaned mask pixel for pixel. Undo recovered the complete original annotation, including editing controls. Redo and reload retained the cleaned result; another cleanup was an exact no-op.

The original project was restored after testing. All seven original image states and project metadata remained equal to the pre-test baseline. Local evidence: `output/qa/hole-cleanup/verification.json`, `target-before.json`, `target-cleaned.json`, and screenshots `output/playwright/hole-cleanup-before.png` / `hole-cleanup-after.png`.

## Visual examples · October 2, 2026

SAM 3's native exemplar API takes boxes in the image being segmented. [Meta's response on cross-image reuse](https://github.com/facebookresearch/sam3/issues/183) states that it is not officially supported. The new experimental path encodes a box in a separate reference image, combines its geometry tokens with the optional text tokens, and supplies them to the detector for the target image. It uses the installed Transformers 5.18 components and official weights, without hooks, model mutation, extra models, or a reference/target collage.

On the M4 Max with real MPS inference, native box prompting and the transferred-token path on the **same image** produced identical raw masks, boxes, logits, and presence logits: maximum absolute difference **0**. This verifies the assembly of the prompt tokens on this example, not general cross-image recognition quality.

For genuinely different source images, a carrot in `sample/images/1047.jpg`, selected with `[38, 57, 170, 119]`, produced **9 proposals** on `1001.jpg`; adding `carrot` produced **20**. Visual inspection found carrot regions with some fragmented boundaries. The cached visual-only request took 142 ms at the engine level in one run, excluding HTTP, contour extraction, and rendering; this is not a latency distribution.

**Quality controls exposed failures.** A potato from `1190.jpg`, selected with `[83, 25, 158, 91]`, produced **8 false positives** on the carrot image. A tightly cropped carrot upload with a full-image box selected background at approximately 99% confidence. Padding the reference did not reliably fix this and is not part of the engine. The interface therefore requires an explicit object box, displays the experimental limitation, and never accepts proposals automatically. Box selection alone does not guarantee correct concept recognition.

Click refinement completed using both cached detector logits and a persisted binary RLE. Both returned masks at the target's 224 × 224 resolution. In one demanding adjacent-click case, the cached path included the positive point and excluded the negative; the RLE-seeded path did not exclude the negative. Those outputs differed by 4,850 pixels (agreement IoU 0.259, not accuracy against ground truth). This is a limitation of continued model refinement from a binary seed; reopening without new inference preserves the saved mask exactly.

Chrome checks exercised upload, required box selection, visual-only inference, accepting two proposals together, refining a third with positive/negative clicks, Enter confirmation, image navigation, reload, and COCO export. All three saved annotations and the six remaining proposals were recovered unchanged; COCO reconstructed the annotation masks pixel for pixel. A subsequent reference + Spanish `zanahoria` search displayed `carrot`, produced 20 proposals, and retained the three existing annotations. The reference remained available across image navigation and cleared on reload as documented. The original sample project files remained unchanged. At widths of 1024, 1440, and 1920 pixels, the prompt stayed outside the canvas without horizontal page overflow.

Automated checks cover invalid/corrupt/animated files, raw base64 and byte/pixel limits (including streamed requests), EXIF orientation, RGB/RGBA/LA/palette transparency, crop coordinates, class/project isolation, reference-only and combined prompts, target-only refinement caches, long prompts, canceled uploads, old responses after changing references or projects, modal keyboard isolation, and ordinary proposal acceptance. Transparent pixels are composited onto the same white background used in the preview.

Evidence: `output/qa/visual-reference/report.json`, `review.json`, overlays and RLE manifests; `output/qa/visual-ui/verification.json`, saved states and exported COCO; screenshots under `output/playwright/visual-*.png`. The model report separates successful execution from failed quality controls. Reproduce with `scripts/benchmark_visual.py --help`; its inputs are user-provided images, not fixtures shipped in the repository. This new inference path was exercised on MPS; CPU/CUDA/WSL behavior has not been revalidated for it.

## Hosted mode and ARM64 CUDA · October 7, 2026

The separate hosted backend and frontend passed the existing Python suite plus
new ownership, session/CSRF, signed OIDC token, storage failure, queue, worker,
and maintenance tests: **263 Python tests and 65 frontend tests passed**, and the
production frontend build passed. Six PostgreSQL checks are opt-in and skipped
in the default Python command; all six passed separately against PostgreSQL 15.
Those integration checks exercised simultaneous quota
reservations, job admission/cancellation, parallel schema initialization, and
a real `pg_dump`/`pg_restore` round trip coordinated with project SQLite snapshots.
The backup race test waited behind an in-flight writer and recovered matching
metadata and annotations. These tests use isolated storage and test credentials.
Storage accounting includes thumbnails and canonical annotation/project data.
Fault injection around journal reservation and atomic SQLite replacement verifies
that accepted mutations recover once and rejected writes preserve the live state.

Real SAM 3 weights ran on NVIDIA GB10 ARM64 with CUDA. After model
warmup, the worker processed points, a box, a polygon seed, English text, Spanish text,
and a visual reference, returning masks with the expected image dimensions.
Individual requests on the synthetic 320 × 240 fixture took approximately
0.2–0.7 seconds with the final pinned container. This is a runtime compatibility check, not an accuracy or
production latency benchmark. Private machine names, network addresses, and
credentials are intentionally excluded from this report.

A real dispatcher integration recovered an accepted attempt after a dispatcher
restart, rejected duplicate execution, and completed a job on the fallback while
the primary was stopped. In a separate accepted-attempt failure, it waited for
the original 120-second deadline plus five seconds before a successful second
attempt, charging the logical request only once. Only isolated annotation
services and disposable integration metadata were used for these checks.

Chrome exercised project creation, reopening, class creation, a new manual
polygon with the worker unavailable, confirmation, saving, and COCO download.
The downloaded COCO contained the expected original filename, one class, and
one annotation with a 240 × 320 RLE. Browser file selection was blocked by the
test browser extension's file URL permission; that fixture image was uploaded
through the authenticated API. Automated tests separately cover file/folder
upload UI behavior and upload validation.

The hosted editor now offers **Use polygon**, which rasterizes a closed polygon
on CPU and permits confirmation without a GPU request or inference quota.
The original local entry point and its existing behavior remain separate.

An isolated localhost service using the unchanged hosted authentication router
and PostgreSQL 15 completed real Google consent, authorization-code exchange,
signed identity-token validation, logout, and a second login to the same account.
Rejecting consent created no account or session. Expiring that session in the
isolated database caused the browser to return to the signed-out state. Separate
controlled checks covered invalid/expired state, replay rejection, and PKCE.
Only basic identity scopes were configured; no credentials are stored in Git.

The CPU Docker image built successfully on Linux x86-64. These checks do not
establish production Google OAuth, R2 credential isolation, public HTTPS, or public
DNS readiness; those require the actual external configuration and launch tests.

## Real Cloudflare R2 and PostgreSQL restore · October 7, 2026

Both environment-specific R2 credentials passed object write, read, list, and
delete checks in their own bucket. Every cross-bucket read, write, and list
attempt returned **HTTP 403 `AccessDenied`**. Unsigned object reads against the
S3 endpoint were rejected with **HTTP 400 `InvalidArgument`**. The checks used
unique disposable prefixes; no deployment objects or databases were modified.

Authenticated application routes ran through FastAPI TestClient with real R2,
PostgreSQL 15, and the deployed GPU worker. Upload, CPU polygon annotation,
save/reopen, COCO export/import, revision conflicts, CSRF rejection, and owner
isolation passed. One normal points request completed in one attempt, using one
quota unit, on a synthetic 256 × 192 image; its observed 1.65-second elapsed time
is a single integration sample, not a production latency benchmark. Test sessions
were created only in disposable databases, without invoking Google OAuth.

A coordinated PostgreSQL/SQLite backup was written to R2 and restored into an
empty database and volume after deleting the source project. Original image
bytes and saved annotation masks were recovered, and old sessions were
invalidated. Corrupting a disposable backup artifact caused checksum rejection
before any destination database mutation. With the maintenance clock advanced
to six days, the deleted original remained protected by the retained manifest;
at eight days, the expired backup and original were collected. All temporary
R2 objects and test databases were removed after verification.

These storage and API checks do not exercise browser Google consent, public
DNS/TLS, or the separate R2 public-domain settings. Those launch checks are
recorded independently when completed.

## Public HTTPS annotation workflow · October 7, 2026

The production Google OAuth app is published with an External audience; Google
reports that the three basic identity scopes do not require verification.
Branding approval remains pending domain ownership verification and Google's
stated 24-hour propagation window.

Chrome completed Google sign-in on the public HTTPS deployment with the
authorized test account. In the real editor, Spanish `zanahorias` produced
19 SAM 3 proposals. One was accepted and saved; reloading and reopening the
project recovered that annotation and the 18 remaining proposals. COCO export
downloaded a dataset containing seven images and one annotation.

A read-only comparison decoded the downloaded COCO RLE and the annotation in
the production project's SQLite database. Both masks were 224 × 224 with
1,994 foreground pixels and **zero differing pixels**. The saved image revision
contained one annotation and 18 proposals. The project and its images were
retained for continued use.

Automated browser file selection remained blocked by the browser extension's
file URL permission. The test image was uploaded through the authenticated
public HTTPS API, which returned HTTP 201. Its temporary test session was
revoked while existing browser sessions remained valid. A subsequent read-only
check confirmed exactly one copy of the fixture, valid original bytes in R2,
and matching active-data quota accounting. This is not a completed browser
file-picker test.

Public edge checks confirmed HTTP-to-HTTPS redirection, HTTP 200 for the home
and legal pages, and HTTP 401 with `private, no-store` for protected anonymous
API requests. The OAuth flow cookie carried Secure, HttpOnly, and SameSite=Lax;
PKCE S256, state, nonce, exact identity scopes, and the configured callback were
present. Duplicate `nosniff` headers from the application and Nginx were corrected
in the site and deployment template; Nginx validation and repeated public-header
checks passed.

## Remaining validation

Segmentation quality still needs evaluation on representative user images, along with prolonged annotation sessions. Translation should be reviewed for the intended domain and wording.

ARM64 Linux/CUDA has now passed the hosted runtime checks above. Windows/WSL and
the x86-64 GPU installer have not been physically revalidated in this phase.
