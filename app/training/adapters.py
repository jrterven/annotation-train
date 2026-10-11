"""Pinned model adapters. Imported only inside the isolated training subprocess."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import time

from PIL import Image
from .data import encoded
from .catalog import model_profile

LOCK = json.loads(Path(__file__).with_name("runtime-lock.json").read_text())


def quiet_environment():
    os.environ.update(
        WANDB_MODE="disabled",
        WANDB_DISABLED="true",
        HF_HUB_DISABLE_TELEMETRY="1",
        DO_NOT_TRACK="1",
        YOLO_AUTOINSTALL="false",
        YOLO_VERBOSE="false",
    )


def verify_runtime(model_id, weights_dir):
    profile = model_profile(model_id)
    package = "ultralytics" if profile["family"] == "yolo26" else "rfdetr"
    if version(package) != LOCK["packages"][package]:
        raise ValueError(
            f"Install {package}=={LOCK['packages'][package]} in the training environment."
        )
    item = LOCK["weights"][model_id]
    path = Path(weights_dir) / item["file"]
    with path.open("rb") as source:
        if hashlib.file_digest(source, item["algorithm"]).hexdigest() != item["checksum"]:
            raise ValueError("Pretrained weights do not match the pinned catalog checksum.")
    return path


def prepare_dataset(manifest, originals, destination, family):
    """Synthetic filenames avoid collisions; test images are absent from fit assignments."""
    from app.yolo import polygon

    destination = Path(destination)
    indices = {c["category_id"]: c["index"] for c in manifest["classes"]}
    for split in {im.get("split", "test") for im in manifest["images"]}:
        folder = destination / (
            {"validation": "valid"}.get(split, split) if family == "rfdetr" else "images/" + split
        )
        folder.mkdir(parents=True, exist_ok=True)
        coco = {
            "info": {},
            "images": [],
            "annotations": [],
            "categories": [
                {"id": c["index"], "name": c["name"], "supercategory": "none"}
                for c in manifest["classes"]
            ],
        }
        for im in manifest["images"]:
            if im.get("split", "test") != split:
                continue
            name = f"{im['id']}.png"
            with Image.open(Path(originals) / im["sha256"]) as source:
                if source.size != (im["width"], im["height"]):
                    raise ValueError("Original dimensions do not match the snapshot")
                clean = source.convert("RGB")
                clean.info.clear()
                clean.save(folder / name)
            coco["images"].append(
                {"id": im["id"], "file_name": name, "width": im["width"], "height": im["height"]}
            )
            labels = []
            for ann in im["annotations"]:
                if manifest["task"] == "detection":
                    x, y, w, h = ann["bbox"]
                    coords = [
                        (x + w / 2) / im["width"],
                        (y + h / 2) / im["height"],
                        w / im["width"],
                        h / im["height"],
                    ]
                    record = {"bbox": [x, y, w, h], "area": w * h}
                else:
                    from app.geometry import decode_mask
                    from pycocotools import mask as mask_utils

                    record = {
                        "segmentation": ann["mask"],
                        "area": int(decode_mask(ann["mask"]).sum()),
                        "bbox": mask_utils.toBbox(
                            {**ann["mask"], "counts": ann["mask"]["counts"].encode()}
                        ).tolist(),
                    }
                    coords = (
                        [
                            n / d
                            for point in polygon(ann["components"])
                            for n, d in zip(point, (im["width"], im["height"]))
                        ]
                        if family == "yolo26"
                        else []
                    )
                labels.append(
                    str(indices[ann["category_id"]])
                    + " "
                    + " ".join(format(n, ".12g") for n in coords)
                )
                coco["annotations"].append(
                    {
                        "id": len(coco["annotations"]) + 1,
                        "image_id": im["id"],
                        "category_id": indices[ann["category_id"]],
                        "iscrowd": ann.get("iscrowd", 0),
                        **record,
                    }
                )
            if family == "yolo26":
                label_dir = destination / "labels" / split
                label_dir.mkdir(parents=True, exist_ok=True)
                (label_dir / f"{im['id']}.txt").write_text("\n".join(labels))
        if family == "rfdetr":
            (folder / "_annotations.coco.json").write_bytes(encoded(coco))
    if family == "yolo26":
        import yaml

        configuration = {
            "path": str(destination.resolve()),
            "train": "images/train",
            "val": "images/validation",
            "names": {c["index"]: c["name"] for c in manifest["classes"]},
        }
        # Deliberately no test entry during training, even when a snapshot has test.
        (destination / "data.yaml").write_text(yaml.safe_dump(configuration))
    return destination


def scalar(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def total_loss(values):
    """Ultralytics 8.4 uses named tensor losses; omit non-finite samples."""
    if values is None:
        return None
    samples = [scalar(value) for value in values.values()]
    return sum(samples) if all(value is not None for value in samples) else None


def train(work, manifest, directory, weights_dir, emit, paused):
    quiet_environment()
    if any(im.get("split") not in {"train", "validation"} for im in manifest["images"]):
        raise ValueError("Training must never receive test images")
    profile, params = work["model"], work["parameters"]
    original_weights = verify_runtime(profile["id"], weights_dir)
    data = prepare_dataset(
        manifest, directory / "originals", directory / "dataset", profile["family"]
    )
    started = time.monotonic() - (
        work.get("metrics", [{}])[-1].get("elapsed", 0) if work.get("metrics") else 0
    )
    output = directory / "fit"
    output.mkdir(exist_ok=True)
    if profile["family"] == "yolo26":
        from ultralytics import YOLO, settings
        from ultralytics.data import utils as data_utils

        # Plotting is disabled. Avoid the dataset checker downloading a font.
        data_utils.check_font = lambda *args, **kwargs: None
        settings.update(
            {
                key: False
                for key in (
                    "sync",
                    "wandb",
                    "comet",
                    "clearml",
                    "mlflow",
                    "neptune",
                    "tensorboard",
                    "dvc",
                    "hub",
                )
                if key in settings
            }
        )
        resume = directory / "resume.pt"
        model = YOLO(str(resume if work.get("resume") else original_weights))

        def epoch(trainer):
            if not Path(trainer.last).is_file() or trainer.epoch + 1 > params["epochs"]:
                return
            metrics = trainer.metrics
            suffix = "M" if profile["task"] == "segmentation" else "B"
            loss = total_loss(trainer.tloss)
            metric = {
                "epoch": trainer.epoch + 1,
                "train_loss": loss,
                "validation_loss": sum(v for k, v in metrics.items() if k.startswith("val/")),
                "map": scalar(metrics.get(f"metrics/mAP50-95({suffix})")),
                "box_map": scalar(metrics.get("metrics/mAP50-95(B)")),
                "precision": scalar(metrics.get(f"metrics/precision({suffix})")),
                "recall": scalar(metrics.get(f"metrics/recall({suffix})")),
                "elapsed": time.monotonic() - started,
            }
            validator_metrics = trainer.validator.metrics
            class_metrics = (
                validator_metrics.seg
                if profile["task"] == "segmentation"
                else validator_metrics.box
            )
            metric["per_class"] = [
                {
                    "category_id": manifest["classes"][int(ci)]["category_id"],
                    "map": float(class_metrics.ap[i]),
                    "precision": float(class_metrics.p[i]),
                    "recall": float(class_metrics.r[i]),
                }
                for i, ci in enumerate(class_metrics.ap_class_index)
            ]
            shutil.copyfile(trainer.last, directory / "resume.next.pt")
            os.replace(directory / "resume.next.pt", resume)
            emit({"metric": {k: v for k, v in metric.items() if v is not None}, "checkpoint": True})
            if paused():
                trainer.stop = True

        # final_eval fires on_fit_epoch_end again *after* stripping the optimizer
        # from last.pt. Only copy checkpoints at the actual save boundary.
        model.add_callback("on_model_save", epoch)

        def batch(trainer):
            emit({"progress": min(1, (trainer.epoch + 0.5) / params["epochs"])})

        model.add_callback("on_train_batch_end", batch)
        kwargs = {
            "data": str(data / "data.yaml"),
            "epochs": params["epochs"],
            "patience": params["patience"],
            "imgsz": params["resolution"],
            "batch": params["batch"],
            "device": 0,
            "workers": 2,
            "project": str(directory),
            "name": "fit",
            "exist_ok": True,
            "seed": manifest["options"]["seed"],
            # AMP's automatic compatibility probe downloads an additional,
            # unpinned model. Use full precision in this runtime profile.
            "amp": False,
            "plots": False,
            "save": True,
            "val": True,
            "verbose": False,
        }
        if params["learning_rate"] is not None:
            kwargs.update(lr0=params["learning_rate"], optimizer="AdamW")
        if not params["augment"]:
            kwargs.update(
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.0,
                degrees=0.0,
                translate=0.0,
                scale=0.0,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.0,
                mosaic=0.0,
                mixup=0.0,
                copy_paste=0.0,
                cutmix=0.0,
            )
        if work.get("resume"):
            kwargs["resume"] = True
        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics.models.yolo.segment import SegmentationTrainer

        base = SegmentationTrainer if profile["task"] == "segmentation" else DetectionTrainer

        class PrimaryMetricTrainer(base):
            def validate(self):
                previous = self.best_fitness
                metrics, _ = super().validate()
                if metrics is None:
                    return None, None
                suffix = "M" if profile["task"] == "segmentation" else "B"
                fitness = float(metrics[f"metrics/mAP50-95({suffix})"])
                self.best_fitness = fitness if previous is None else max(previous, fitness)
                return metrics, fitness

        model.train(trainer=PrimaryMetricTrainer, **kwargs)
        effective = {
            k: v
            for k, v in vars(model.trainer.args).items()
            if k
            not in {"data", "model", "project", "name", "save_dir", "device", "source", "resume"}
        }
        best = output / "weights" / "best.pt"
        target = directory / "model.pt"
    else:
        import rfdetr
        from pytorch_lightning import Callback
        from rfdetr.training import RFDETRModelModule, RFDETRDataModule, build_trainer

        name = (
            "RFDETR"
            + ("Seg" if profile["task"] == "segmentation" else "")
            + profile["size"].title()
        )
        model = getattr(rfdetr, name)(pretrain_weights=str(original_weights), device="cuda")
        kwargs = {
            "dataset_dir": str(data),
            "output_dir": str(output),
            "epochs": params["epochs"],
            "batch_size": params["batch"],
            "grad_accum_steps": 1,
            "num_workers": 2,
            "early_stopping": True,
            "early_stopping_patience": params["patience"],
            "early_stopping_min_delta": 0.0,
            "run_test": False,
            "tensorboard": False,
            "wandb": False,
            "mlflow": False,
            "clearml": False,
            "progress_bar": None,
            "save_dataset_grids": False,
            "compute_val_loss": True,
            "log_per_class_metrics": True,
            "seed": manifest["options"]["seed"],
            "eval_masks_head_resolution": False,
        }
        if params["learning_rate"] is not None:
            kwargs["lr"] = params["learning_rate"]
        if not params["augment"]:
            kwargs.update(aug_config={}, scale_jitter=False, multi_scale=False)
        config = model.get_train_config(**kwargs)
        effective = config.model_dump(mode="json", exclude={"dataset_dir", "output_dir", "resume"})
        model._align_num_classes_from_dataset(str(data))
        module = RFDETRModelModule(model.model_config, config)
        datamodule = RFDETRDataModule(model.model_config, config)
        trainer = build_trainer(config, model.model_config, accelerator="gpu", devices=[0])
        resume = directory / "resume.ckpt"
        validation_classes = []
        # The pinned RF-DETR callback logs per-class bbox AP even for a mask
        # model. Capture its already-computed mask AP before the metric resets.
        for callback in trainer.callbacks:
            if type(callback).__name__ != "COCOEvalCallback":
                continue
            original_compute = callback._compute_map_metric

            def capture_metrics(trainer, metric, original=original_compute, callback=callback):
                values = original(trainer, metric)
                if metric is callback.map_metric or metric is callback.map_metric_ema:
                    prefix = "segm_" if profile["task"] == "segmentation" else ""
                    validation_classes.clear()
                    for index, ap in zip(
                        values.get("classes", []), values.get(prefix + "map_per_class", [])
                    ):
                        value = scalar(ap)
                        if value is not None and value >= 0:
                            validation_classes.append(
                                {
                                    "category_id": manifest["classes"][int(index)]["category_id"],
                                    "map": value,
                                }
                            )
                return values

            callback._compute_map_metric = capture_metrics

        class Progress(Callback):
            def on_train_epoch_end(self, trainer, pl_module):
                metrics = trainer.callback_metrics
                key = "val/segm_mAP_50_95" if profile["task"] == "segmentation" else "val/mAP_50_95"
                metric = {
                    "epoch": trainer.current_epoch + 1,
                    "train_loss": scalar(
                        metrics.get("train/loss_epoch", metrics.get("train/loss"))
                    ),
                    "validation_loss": scalar(metrics.get("val/loss")),
                    "map": scalar(metrics.get(key)),
                    "box_map": scalar(metrics.get("val/mAP_50_95")),
                    "precision": scalar(metrics.get("val/precision")),
                    "recall": scalar(metrics.get("val/recall")),
                    "elapsed": time.monotonic() - started,
                }
                metric["per_class"] = list(validation_classes)
                trainer.save_checkpoint(str(directory / "resume.next.ckpt"))
                os.replace(directory / "resume.next.ckpt", resume)
                emit(
                    {
                        "metric": {k: v for k, v in metric.items() if v is not None},
                        "checkpoint": True,
                    }
                )
                if paused():
                    trainer.should_stop = True

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                emit(
                    {
                        "progress": min(
                            1,
                            (
                                trainer.current_epoch
                                + (batch_idx + 1) / max(1, trainer.num_training_batches)
                            )
                            / params["epochs"],
                        )
                    }
                )

        trainer.callbacks.append(Progress())
        trainer.fit(module, datamodule, ckpt_path=str(resume) if work.get("resume") else None)
        best = output / "checkpoint_best_total.pth"
        target = directory / "model.pth"
    if paused():
        return {"status": "paused", "checkpoint": True}
    if not best.is_file():
        raise RuntimeError("Training finished without a best validation checkpoint")
    shutil.copyfile(best, target)
    with original_weights.open("rb") as source:
        pretrained_sha = hashlib.file_digest(source, "sha256").hexdigest()
    (directory / "manifest.json").write_bytes(
        encoded(
            {
                "model": profile,
                "parameters": params,
                "dataset_fingerprint": manifest.get("fingerprint"),
                "classes": manifest["classes"],
                "runtime": work["runtime"],
                "effective_parameters": effective,
                "pretrained_sha256": pretrained_sha,
                "checkpoint_selection": "validation",
            }
        )
    )
    return {"status": "syncing", "checkpoint": True, "model_file": target.name}


def predictor(work, checkpoint):
    quiet_environment()
    profile = work["model"]
    if profile["family"] == "yolo26":
        from ultralytics import YOLO

        model = YOLO(str(checkpoint))

        def infer(path, confidence):
            result = model.predict(
                str(path), conf=confidence, device=0, retina_masks=True, verbose=False
            )[0]
            return (
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
                result.boxes.conf.cpu().numpy(),
                (result.masks.data.cpu().numpy() > 0.5 if result.masks is not None else None),
            )

    else:
        import rfdetr

        name = (
            "RFDETR"
            + ("Seg" if profile["task"] == "segmentation" else "")
            + profile["size"].title()
        )
        model = getattr(rfdetr, name)(pretrain_weights=str(checkpoint), device="cuda")

        def infer(path, confidence):
            result = model.predict(str(path), threshold=confidence)
            # RF-DETR exposes its no-object logit at very low confidence.
            # It is background, never a user category or proposal.
            result = result[result.data["class_name"] != "__background__"]
            return result.xyxy, result.class_id, result.confidence, result.mask

    return infer


def predict(work, manifest, directory):
    from app.geometry import mask_payload

    infer = predictor(
        work, directory / ("model.pt" if work["model"]["family"] == "yolo26" else "model.pth")
    )
    classes = {c["index"]: c["category_id"] for c in manifest["classes"]}
    results = []
    for image in manifest["images"]:
        path = directory / f"predict-{image['id']}.png"
        with Image.open(directory / "originals" / image["sha256"]) as original:
            clean = original.convert("RGB")
            clean.info.clear()
            clean.save(path)
        boxes, labels, scores, masks = infer(path, work.get("confidence", 0.001))
        proposals = []
        for index, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            if int(label) not in classes:
                raise ValueError("Model class mapping differs from the saved manifest")
            x1, y1, x2, y2 = map(float, box)
            value = {
                "category_id": classes[int(label)],
                "score": float(score),
                "iscrowd": 0,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
            }
            if manifest["task"] == "segmentation":
                if masks is None or masks[index].shape != (image["height"], image["width"]):
                    raise ValueError("Model did not return an original-resolution mask")
                value.update(mask_payload(masks[index].astype(bool), include_preview=False))
            else:
                value["kind"] = "bbox"
            proposals.append(value)
        results.append({"image_id": image["id"], "proposals": proposals})
        path.unlink(missing_ok=True)
    return {"images": results}


def evaluate(work, manifest, directory):
    if any(image.get("split") != "test" for image in manifest["images"]):
        raise ValueError("Final evaluation accepts only reserved test images")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mu

    result = predict({**work, "confidence": 0.001}, manifest, directory)
    ground = {
        "info": {},
        "images": [],
        "categories": [{"id": c["category_id"], "name": c["name"]} for c in manifest["classes"]],
        "annotations": [],
    }
    for image in manifest["images"]:
        if image["split"] != "test":
            raise ValueError("Final evaluation accepts only reserved test images")
        ground["images"].append({key: image[key] for key in ("id", "width", "height")})
        for ann in image["annotations"]:
            entry = {
                "id": len(ground["annotations"]) + 1,
                "image_id": image["id"],
                "category_id": ann["category_id"],
                "iscrowd": ann.get("iscrowd", 0),
            }
            if manifest["task"] == "segmentation":
                rle = {**ann["mask"], "counts": ann["mask"]["counts"].encode()}
                entry.update(
                    segmentation=rle, area=float(mu.area(rle)), bbox=mu.toBbox(rle).tolist()
                )
            else:
                entry.update(bbox=ann["bbox"], area=ann["bbox"][2] * ann["bbox"][3])
            ground["annotations"].append(entry)
    gt = COCO()
    gt.dataset = ground
    gt.createIndex()
    detections = []
    boxes = []
    for image in result["images"]:
        for proposal in image["proposals"]:
            item = {
                "image_id": image["image_id"],
                "category_id": proposal["category_id"],
                "score": proposal["score"],
            }
            if manifest["task"] == "segmentation":
                item["segmentation"] = {
                    **proposal["mask"],
                    "counts": proposal["mask"]["counts"].encode(),
                }
            else:
                item["bbox"] = proposal["bbox"]
            detections.append(item)
            boxes.append(
                {
                    "image_id": image["image_id"],
                    "category_id": proposal["category_id"],
                    "score": proposal["score"],
                    "bbox": proposal["bbox"],
                }
            )
    if detections:
        dt = gt.loadRes(detections)
    else:
        dt = COCO()
        dt.dataset = {**ground, "annotations": []}
        dt.createIndex()
    evaluator = COCOeval(gt, dt, "segm" if manifest["task"] == "segmentation" else "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    metrics = {
        key: float(evaluator.stats[i])
        for i, key in enumerate(("mAP50-95", "AP50", "AP75"))
        if evaluator.stats[i] >= 0
    }
    per_class = []
    for index, category_id in enumerate(evaluator.params.catIds):
        precision = evaluator.eval["precision"][:, :, index, 0, -1]
        valid = precision[precision >= 0]
        if valid.size:
            per_class.append({"category_id": int(category_id), "map": float(valid.mean())})
    if manifest["task"] == "segmentation":
        box_dt = gt.loadRes(boxes) if boxes else dt
        box_evaluator = COCOeval(gt, box_dt, "bbox")
        box_evaluator.evaluate()
        box_evaluator.accumulate()
        box_evaluator.summarize()
        if box_evaluator.stats[0] >= 0:
            metrics["box-mAP50-95"] = float(box_evaluator.stats[0])
    return {
        "metrics": metrics,
        "per_class": per_class,
        "snapshot_id": work["dataset_id"],
        "checkpoint_sha256": work["checkpoint_sha256"],
        "partition": "test",
    }
