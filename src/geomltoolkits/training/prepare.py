import json
import os
from pathlib import Path

import numpy as np
import rasterio

from .._logging import get_logger

log = get_logger(__name__)


def prepare_dataset(
    chips_dir: str,
    labels_dir: str,
    output_dir: str,
    dataset_format: str = "yolo",
    class_names: list[str] | None = None,
) -> str:
    """Match chips to burned labels by filename and emit dataset metadata.

    Produces train-only output (no val/test splitting). Supports YOLO yaml
    and COCO json formats.
    """
    if class_names is None:
        class_names = ["building"]

    os.makedirs(output_dir, exist_ok=True)

    pairs = _match_chips_and_labels(chips_dir, labels_dir)
    log.info(f"Matched {len(pairs)} chip-label pairs")

    images_dir = os.path.join(output_dir, "images", "train")
    masks_dir = os.path.join(output_dir, "labels", "train")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    for chip_path, label_path in pairs:
        chip_name = os.path.basename(chip_path)
        label_name = os.path.basename(label_path)
        _symlink_or_copy(chip_path, os.path.join(images_dir, chip_name))
        _symlink_or_copy(label_path, os.path.join(masks_dir, label_name))

    if dataset_format == "yolo":
        _emit_yolo_yaml(output_dir, class_names)
    elif dataset_format == "coco":
        _emit_coco_json(output_dir, pairs, class_names)
    else:
        raise ValueError(f"Unsupported format: {dataset_format}. Use 'yolo' or 'coco'.")

    log.info(f"Dataset prepared in {output_dir} ({dataset_format} format)")
    return output_dir


def _match_chips_and_labels(chips_dir: str, labels_dir: str) -> list[tuple[str, str]]:
    chip_stems = {Path(f).stem: os.path.join(chips_dir, f) for f in os.listdir(chips_dir) if _is_raster(f)}
    label_stems = {Path(f).stem: os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if _is_raster(f)}

    common = sorted(chip_stems.keys() & label_stems.keys())
    return [(chip_stems[stem], label_stems[stem]) for stem in common]


def _is_raster(filename: str) -> bool:
    return filename.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))


def _symlink_or_copy(src: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def _emit_yolo_yaml(output_dir: str, class_names: list[str]) -> None:
    yaml_content = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "",
        "names": {i: name for i, name in enumerate(class_names)},
    }

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    import yaml

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    log.info(f"YOLO dataset.yaml written to {yaml_path}")


def _emit_coco_json(output_dir: str, pairs: list[tuple[str, str]], class_names: list[str]) -> None:
    categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]
    images = []
    annotations = []
    annotation_id = 0

    for img_id, (chip_path, label_path) in enumerate(pairs):
        with rasterio.open(chip_path) as src:
            width, height = src.width, src.height

        images.append(
            {
                "id": img_id,
                "file_name": os.path.basename(chip_path),
                "width": width,
                "height": height,
            }
        )

        with rasterio.open(label_path) as src:
            mask = src.read(1)

        if np.any(mask > 0):
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": 0,
                    "bbox": _mask_to_bbox(mask),
                    "area": int(np.sum(mask > 0)),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    coco_path = os.path.join(output_dir, "annotations.json")
    with open(coco_path, "w") as f:
        json.dump(coco, f)

    log.info(f"COCO annotations.json written to {coco_path}")


def _mask_to_bbox(mask: np.ndarray) -> list[int]:
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]
