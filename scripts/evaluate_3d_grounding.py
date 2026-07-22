#!/usr/bin/env python3
"""Distributed deterministic evaluation of the official native-3D cookbook."""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "three_d_grounding"
IMAGES = ROOT / "cookbooks" / "assets" / "3d_grounding" / "images"
RESULTS = Path("/tmp/rynnbrain-3d-results")
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


def build_prompt(category: str, intrinsics: list[float]) -> str:
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
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    config = json.loads((ROOT / "config.json").read_text())
    cases = json.loads((DATA / "test_cases.json").read_text())
    if world != len(cases):
        raise RuntimeError(f"expected {len(cases)} ranks, received {world}")

    model_dir = Path("/tmp/model-snapshot")
    if rank == 0:
        RESULTS.mkdir(parents=True, exist_ok=True)
        print(f"Downloading immutable snapshot for {config['model_id']}", flush=True)
        snapshot_download(repo_id=config["model_id"], local_dir=model_dir, max_workers=16)
    dist.barrier()

    case = cases[rank]
    image_path = IMAGES / case["image"]
    intrinsics_scale = float(config.get("intrinsics_scale", 1.0))
    principal_point_dx = float(config.get("principal_point_dx", 0.0))
    principal_point_dy = float(config.get("principal_point_dy", 0.0))
    prompt_intrinsics = list(case["intrinsics"])
    prompt_intrinsics[0] *= intrinsics_scale
    prompt_intrinsics[1] *= intrinsics_scale
    prompt_intrinsics[2] += principal_point_dx
    prompt_intrinsics[3] += principal_point_dy
    horizontal_flip = bool(config.get("horizontal_flip", False))
    vertical_flip = bool(config.get("vertical_flip", False))
    model_image_path = image_path
    with Image.open(image_path) as source_image:
        image_size = source_image.size
        if horizontal_flip:
            model_image_path = RESULTS / f"horizontal-flip-rank-{rank}.png"
            source_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(
                model_image_path
            )
            prompt_intrinsics[2] = image_size[0] - 1 - prompt_intrinsics[2]
        elif vertical_flip:
            model_image_path = RESULTS / f"vertical-flip-rank-{rank}.png"
            source_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(
                model_image_path
            )
            prompt_intrinsics[3] = image_size[1] - 1 - prompt_intrinsics[3]
    result: dict[str, object] = {
        "rank": rank,
        "case_id": case["id"],
        "category": case["category"],
        "model_id": config["model_id"],
    }
    started = time.perf_counter()
    try:
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval()
        prompt = build_prompt(case["category"], prompt_intrinsics)
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(model_image_path)},
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
        ).to(device)
        torch.cuda.synchronize(device)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(config["max_new_tokens"]),
                do_sample=False,
            )
        torch.cuda.synchronize(device)
        output_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        response = processor.decode(output_ids[0], skip_special_tokens=True)
        boxes = parse_boxes(response)
        physical = bool(boxes and all(box_is_physical(box) for box in boxes))
        projected = bool(
            boxes
            and any(
                center_projects_inside(box, prompt_intrinsics, image_size)
                for box in boxes
            )
        )
        result.update({
            "prompt": prompt,
            "response": response,
            "boxes": boxes,
            "intrinsics_scale": intrinsics_scale,
            "principal_point_dx": principal_point_dx,
            "principal_point_dy": principal_point_dy,
            "horizontal_flip": horizontal_flip,
            "vertical_flip": vertical_flip,
            "prompt_intrinsics": prompt_intrinsics,
            "box_count": len(boxes),
            "format_valid": bool(boxes),
            "physical_valid": physical,
            "projection_valid": projected,
            "inference_seconds": time.perf_counter() - inference_started,
            "total_seconds": time.perf_counter() - started,
            "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        })
        reference = case.get("recorded_reference")
        if boxes and reference is not None:
            result["recorded_reference"] = reference
            result["recorded_center_error_m"] = math.dist(boxes[0][:3], reference[:3])
            result["recorded_dimension_error_m"] = math.dist(boxes[0][3:6], reference[3:6])
            result["recorded_angle_mae"] = statistics.fmean(
                abs(boxes[0][index] - reference[index]) for index in range(6, 9)
            )
        baseline = case.get("calibration_baseline")
        if boxes and baseline is not None:
            expected = list(baseline)
            expected[0] = (
                baseline[0] / intrinsics_scale
                - principal_point_dx
                * baseline[2]
                / (case["intrinsics"][0] * intrinsics_scale)
            )
            expected[1] = (
                baseline[1] / intrinsics_scale
                - principal_point_dy
                * baseline[2]
                / (case["intrinsics"][1] * intrinsics_scale)
            )
            if horizontal_flip:
                expected[0] *= -1
            if vertical_flip:
                expected[1] *= -1
            result["calibration_baseline"] = baseline
            result["calibration_expected_center"] = expected[:3]
            result["calibration_expected_x_error_m"] = abs(
                boxes[0][0] - expected[0]
            )
            result["calibration_unflipped_x_error_m"] = abs(
                boxes[0][0] - baseline[0]
            )
            result["calibration_expected_y_error_m"] = abs(
                boxes[0][1] - expected[1]
            )
            result["calibration_unflipped_y_error_m"] = abs(
                boxes[0][1] - baseline[1]
            )
            result["calibration_expected_center_error_m"] = math.dist(
                boxes[0][:3], expected[:3]
            )
            result["calibration_unscaled_center_error_m"] = math.dist(
                boxes[0][:3], baseline[:3]
            )
    except Exception as exc:
        result.update({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "total_seconds": time.perf_counter() - started,
        })

    (RESULTS / f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
    dist.barrier()

    if rank == 0:
        rows = [json.loads((RESULTS / f"rank-{index}.json").read_text()) for index in range(world)]
        for row in rows:
            print("THREED_CASE_RESULT " + json.dumps(row, sort_keys=True), flush=True)
        successes = [row for row in rows if "error" not in row]
        parsed = [row for row in successes if row.get("format_valid")]
        physical = [row for row in successes if row.get("physical_valid")]
        projected = [row for row in successes if row.get("projection_valid")]
        referenced = [row for row in parsed if "recorded_center_error_m" in row]
        calibrated = [
            row for row in parsed if "calibration_expected_center_error_m" in row
        ]
        summary = {
            "model_id": config["model_id"],
            "cases": len(rows),
            "worker_successes": len(successes),
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
            "mean_calibration_expected_center_error_m": statistics.fmean(
                row["calibration_expected_center_error_m"] for row in calibrated
            ) if calibrated else None,
            "mean_calibration_unscaled_center_error_m": statistics.fmean(
                row["calibration_unscaled_center_error_m"] for row in calibrated
            ) if calibrated else None,
            "mean_calibration_expected_x_error_m": statistics.fmean(
                row["calibration_expected_x_error_m"] for row in calibrated
            ) if calibrated else None,
            "mean_calibration_unflipped_x_error_m": statistics.fmean(
                row["calibration_unflipped_x_error_m"] for row in calibrated
            ) if calibrated else None,
            "mean_calibration_expected_y_error_m": statistics.fmean(
                row["calibration_expected_y_error_m"] for row in calibrated
            ) if calibrated else None,
            "mean_calibration_unflipped_y_error_m": statistics.fmean(
                row["calibration_unflipped_y_error_m"] for row in calibrated
            ) if calibrated else None,
            "mean_inference_seconds": statistics.fmean(
                row["inference_seconds"] for row in successes
            ) if successes else None,
            "max_peak_gpu_gib": max(
                (row["peak_gpu_gib"] for row in successes), default=None
            ),
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
        }
        print("THREED_EVAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
        if len(successes) != len(rows):
            raise SystemExit(2)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
