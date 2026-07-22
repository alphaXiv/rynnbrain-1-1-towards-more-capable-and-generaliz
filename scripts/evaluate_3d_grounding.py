#!/usr/bin/env python3
"""Eight-GPU balanced evaluation of the official native-3D cookbook."""

from __future__ import annotations

import json
import math
import re
import statistics
import time
import traceback
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "three_d_grounding"
IMAGES = ROOT / "cookbooks" / "assets" / "3d_grounding" / "images"
TAG_PATTERN = re.compile(
    r"<3D\s+Grounding>\s*(.*?)\s*</3D\s+Grounding>", re.IGNORECASE | re.DOTALL
)
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse_boxes(text: str) -> list[list[float]]:
    boxes: list[list[float]] = []
    for content in TAG_PATTERN.findall(text):
        values = [float(value) for value in NUMBER_PATTERN.findall(content)]
        if len(values) == 9:
            boxes.append(values)
    return boxes


def format_intrinsics(values: list[float]) -> str:
    fx, fy, cx, cy = values
    rows = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    return "[" + ", ".join(
        "[" + ", ".join(f"{value:.2f}" for value in row) + "]" for row in rows
    ) + "]"


def build_prompt(
    category: str,
    intrinsics: list[float],
    serialization_example: list[float] | None = None,
) -> str:
    example = ""
    if serialization_example is not None:
        serialized = ", ".join(f"{value:.2f}" for value in serialization_example)
        example = (
            "\nSerialization example (format only; do not copy its values):\n"
            f"<3D Grounding> {serialized} </3D Grounding>\n"
            "Return only tagged 3D Grounding boxes, never JSON or prose.\n"
        )
    return f"""Find all {category} in this image.

The camera intrinsics matrix is:
{format_intrinsics(intrinsics)}

Predict 3D bounding boxes in the camera coordinate system, where:
- x points to the right
- y points downward
- z points forward

For each object, return:
<3D Grounding> cx, cy, cz, x_size, y_size, z_size, pitch, yaw, roll </3D Grounding>

Definitions:
- cx, cy, cz: 3D coordinates of the box center in the camera coordinate system, in meters
- x_size, y_size, z_size: box dimensions in the box local coordinate system, in meters
- pitch, yaw, roll: normalized rotations in the range [-1, 1], corresponding to [-180, 180] degrees

Constraints:
- x_size >= z_size
- Use meters for cx, cy, cz, x_size, y_size, z_size
- Use normalized values in [-1, 1] for pitch, yaw, roll
{example}
<think>\n\n</think>\n\n"""


def box_is_physical(box: list[float]) -> bool:
    return bool(
        len(box) == 9
        and all(math.isfinite(value) for value in box)
        and box[2] > 0
        and all(value > 0 for value in box[3:6])
        and box[3] >= box[5]
        and all(-1 <= value <= 1 for value in box[6:9])
    )


def center_projects_inside(
    box: list[float], intrinsics: list[float], image_size: tuple[int, int]
) -> bool:
    if box[2] <= 0:
        return False
    fx, fy, cx, cy = intrinsics
    u = fx * box[0] / box[2] + cx
    v = fy * box[1] / box[2] + cy
    width, height = image_size
    return bool(math.isfinite(u) and math.isfinite(v) and 0 <= u < width and 0 <= v < height)


def main() -> None:
    gpu_count = torch.cuda.device_count()
    if gpu_count != 8:
        raise RuntimeError(f"expected 8 visible GPUs, received {gpu_count}")
    input_device = torch.device("cuda:0")
    config = json.loads((ROOT / "config.json").read_text())
    cases = json.loads((DATA / "test_cases.json").read_text())
    model_dir = Path("/tmp/model-snapshot")
    print(f"Downloading immutable snapshot for {config['model_id']}", flush=True)
    snapshot_download(repo_id=config["model_id"], local_dir=model_dir, max_workers=32)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="balanced",
        max_memory={index: "88GiB" for index in range(gpu_count)},
    ).eval()
    print(
        "SHARDED_MODEL_READY "
        + json.dumps({
            "model_id": config["model_id"],
            "gpu_count": gpu_count,
            "device_map": getattr(model, "hf_device_map", None),
        }, sort_keys=True, default=str),
        flush=True,
    )

    rows: list[dict[str, object]] = []
    for rank, case in enumerate(cases):
        image_path = IMAGES / case["image"]
        result: dict[str, object] = {
            "rank": rank,
            "case_id": case["id"],
            "category": case["category"],
            "model_id": config["model_id"],
        }
        started = time.perf_counter()
        try:
            serialization_example = config.get("serialization_example")
            prompt = build_prompt(
                case["category"], case["intrinsics"], serialization_example
            )
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }]
            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(input_device)
            for index in range(gpu_count):
                torch.cuda.reset_peak_memory_stats(index)
                torch.cuda.synchronize(index)
            inference_started = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=int(config["max_new_tokens"]),
                    do_sample=False,
                )
            for index in range(gpu_count):
                torch.cuda.synchronize(index)
            inference_seconds = time.perf_counter() - inference_started
            output_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            response = processor.decode(output_ids[0], skip_special_tokens=True)
            boxes = parse_boxes(response)
            with Image.open(image_path) as image:
                image_size = image.size
            physical = bool(boxes and all(box_is_physical(box) for box in boxes))
            projected = bool(
                boxes
                and any(
                    center_projects_inside(box, case["intrinsics"], image_size)
                    for box in boxes
                )
            )
            result.update({
                "prompt": prompt,
                "response": response,
                "boxes": boxes,
                "serialization_example": serialization_example,
                "box_count": len(boxes),
                "format_valid": bool(boxes),
                "physical_valid": physical,
                "projection_valid": projected,
                "inference_seconds": inference_seconds,
                "total_seconds": time.perf_counter() - started,
                "peak_gpu_gib": max(
                    torch.cuda.max_memory_allocated(index) / (1024**3)
                    for index in range(gpu_count)
                ),
            })
            reference = case.get("recorded_reference")
            if boxes and reference is not None:
                result["recorded_reference"] = reference
                result["recorded_center_error_m"] = math.dist(
                    boxes[0][:3], reference[:3]
                )
                result["recorded_dimension_error_m"] = math.dist(
                    boxes[0][3:6], reference[3:6]
                )
                result["recorded_angle_mae"] = statistics.fmean(
                    abs(boxes[0][index] - reference[index])
                    for index in range(6, 9)
                )
            if boxes and serialization_example is not None:
                result["serialization_anchor_center_error_m"] = math.dist(
                    boxes[0][:3], serialization_example[:3]
                )
                result["serialization_anchor_full_mae"] = statistics.fmean(
                    abs(boxes[0][index] - serialization_example[index])
                    for index in range(9)
                )
        except Exception as exc:
            result.update({
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "total_seconds": time.perf_counter() - started,
            })
        rows.append(result)
        print("THREED_CASE_RESULT " + json.dumps(result, sort_keys=True), flush=True)

    successes = [row for row in rows if "error" not in row]
    parsed = [row for row in successes if row.get("format_valid")]
    physical = [row for row in successes if row.get("physical_valid")]
    projected = [row for row in successes if row.get("projection_valid")]
    referenced = [row for row in parsed if "recorded_center_error_m" in row]
    anchored = [
        row for row in parsed if "serialization_anchor_center_error_m" in row
    ]
    summary = {
        "model_id": config["model_id"],
        "cases": len(rows),
        "worker_successes": len(successes),
        "sharded_gpu_count": gpu_count,
        "format_valid_rate": len(parsed) / len(rows),
        "physical_valid_rate": len(physical) / len(rows),
        "projection_valid_rate": len(projected) / len(rows),
        "mean_box_count": statistics.fmean(row["box_count"] for row in successes),
        "mean_recorded_center_error_m": statistics.fmean(
            row["recorded_center_error_m"] for row in referenced
        ) if referenced else None,
        "mean_recorded_dimension_error_m": statistics.fmean(
            row["recorded_dimension_error_m"] for row in referenced
        ) if referenced else None,
        "mean_recorded_angle_mae": statistics.fmean(
            row["recorded_angle_mae"] for row in referenced
        ) if referenced else None,
        "mean_serialization_anchor_center_error_m": statistics.fmean(
            row["serialization_anchor_center_error_m"] for row in anchored
        ) if anchored else None,
        "mean_serialization_anchor_full_mae": statistics.fmean(
            row["serialization_anchor_full_mae"] for row in anchored
        ) if anchored else None,
        "mean_inference_seconds": statistics.fmean(
            row["inference_seconds"] for row in successes
        ) if successes else None,
        "max_peak_gpu_gib": max(
            (row["peak_gpu_gib"] for row in successes), default=None
        ),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    print("THREED_EVAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if len(successes) != len(rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
