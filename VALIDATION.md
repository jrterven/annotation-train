# Local validation · October 1, 2026

Hardware: MacBook Pro M4 Max, 128 GB, macOS 27.0.1. Isolated Python 3.12.14 environment.

## Completed checks

- `python -m pytest -q`: **142 passing tests** covering persistence, geometry, COCO, API, SAM 3 contracts, polygon drafts, mask-seeded refinement, translation, and explicit hole cleanup.
- `npm run test`: **33 passing tests** covering masks, autosave, conflicts, stale responses, polygon workflows, English interface controls, hole cleanup with undo/redo, and real React Konva rendering.
- `npm run build`: TypeScript compilation and the Vite production build pass.
- `node --experimental-strip-types --test tests/frontend-masks.mjs`: **6 passing tests**. Fixtures encoded by pycocotools are compared pixel by pixel with the browser decoder.
- `pip check`: no incompatible dependencies reported in the current installed environment.
- A clean resolution of `requirements.txt` using `pip --dry-run --ignore-installed --only-binary=:all:` passed for macOS ARM/Python 3.12; report: `output/qa/install-resolution.json`. This check preceded the addition of translation dependencies and does not establish a fresh installation of the updated requirements. Dependencies do not use local filesystem paths.
- PyTorch 2.14.1 and torchvision 0.29.1 import successfully; Transformers 5.18.0 exposes `Sam3Model` and `Sam3TrackerModel`.
- Metal/MPS is available outside the sandbox; a real `torchvision.ops.roi_align` check on MPS tensors passed.
- Pinned SAM checkpoint: `facebook/sam3`, revision `3c879f39826c281e95690f02c7821c4de09afae7`.

Tests cover rectangular masks, holes, islands, single-pixel objects, image borders, manual geometry and vertex edits, optimistic revisions, concurrent saves, COCO IDs, moving projects, external image roots, relinking, and isolation between browser tabs.

Chrome checks include COCO import, image navigation, vertex dragging/insertion/deletion, canvas panning over controls, undo/redo, reload, and export. Undo restored the exact original mask; redo and reload preserved the edit. Both exported COCO files reconstructed pixel-identical masks using `pycocotools.COCO.annToMask`, including the hole. These checks used synthetic test images, not simulated SAM predictions.

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

## Remaining validation

Segmentation quality still needs evaluation on representative user images, along with prolonged annotation sessions. Translation should be reviewed for the intended domain and wording.

Linux/CUDA and Windows/WSL are target platforms supported by the installer design; no physical-system tests were performed on those platforms during this session. A fresh installation of the updated translation dependencies has not been established by the earlier dependency-resolution report.
