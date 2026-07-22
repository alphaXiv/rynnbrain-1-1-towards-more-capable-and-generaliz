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
CALIBRATION_BASELINES = {
    "sunrgbd_chair": [-0.13, 0.01, 1.28],
    "sunrgbd_bed": [-0.11, -0.08, 2.72],
    "sunrgbd_table": [0.23, -0.21, 2.23],
    "office_chairs": [0.29, 0.03, 2.55],
    "lounge_sofa": [-0.90, -0.20, 3.03],
    "lounge_table": [0.19, 0.08, 1.88],
    "manipulation_bottle": [0.01, -0.03, 0.75],
}
SAME_CATEGORY_SWAPS = {
    "sunrgbd_chair": "office_chairs",
    "office_chairs": "sunrgbd_chair",
    "sunrgbd_table": "lounge_table",
    "lounge_table": "sunrgbd_table",
}


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
    include_intrinsics: bool = True,
    explicit_no_object: bool = False,
    serialization_template_only: bool = False,
) -> str:
    intrinsics_block = (
        f"The camera intrinsics matrix is:\n{format_intrinsics(intrinsics)}\n\n"
        if include_intrinsics
        else ""
    )
    absence_instruction = (
        f"If no {category} is visible, return exactly <no object> and do not invent a box.\n\n"
        if explicit_no_object
        else ""
    )
    example = ""
    if serialization_template_only:
        example = (
            "\nFormat template (field names are placeholders, not values):\n"
            "<3D Grounding> cx, cy, cz, x_size, y_size, z_size, "
            "pitch, yaw, roll </3D Grounding>\n"
            "Return only tagged 3D Grounding boxes, never JSON or prose.\n"
        )
    elif serialization_example is not None:
        serialized = ", ".join(f"{value:.2f}" for value in serialization_example)
        example = (
            "\nSerialization example (format only; do not copy its values):\n"
            f"<3D Grounding> {serialized} </3D Grounding>\n"
            "Return only tagged 3D Grounding boxes, never JSON or prose.\n"
        )
    return f"""Find all {category} in this image.

{absence_instruction}{intrinsics_block}Predict 3D bounding boxes in the camera coordinate system, where:
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
    cases_by_id = {case["id"]: case for case in cases}
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
        cyclic_image = bool(config.get("cyclic_image", False))
        same_category_swap = bool(config.get("same_category_swap", False))
        if same_category_swap and case["id"] in SAME_CATEGORY_SWAPS:
            shown_case = cases_by_id[SAME_CATEGORY_SWAPS[case["id"]]]
        else:
            shown_case = cases[(rank + 1) % len(cases)] if cyclic_image else case
        image_path = IMAGES / shown_case["image"]
        swap_intrinsics_with_image = bool(
            config.get("swap_intrinsics_with_image", True)
        )
        intrinsics_case = shown_case if swap_intrinsics_with_image else case
        prompt_intrinsics = list(intrinsics_case["intrinsics"])
        intrinsics_scale = float(config.get("intrinsics_scale", 1.0))
        include_intrinsics = bool(config.get("include_intrinsics", True))
        explicit_no_object = bool(config.get("explicit_no_object", False))
        white_frame = bool(config.get("white_frame", False))
        prompt_intrinsics[0] *= intrinsics_scale
        prompt_intrinsics[1] *= intrinsics_scale
        model_image_path = image_path
        with Image.open(image_path) as source_image:
            image_size = source_image.size
            if white_frame:
                model_image_path = Path(f"/tmp/white-frame-rank-{rank}.png")
                Image.new("RGB", image_size, "white").save(model_image_path)
        result: dict[str, object] = {
            "rank": rank,
            "case_id": case["id"],
            "category": case["category"],
            "shown_case_id": shown_case["id"],
            "shown_category": shown_case["category"],
            "cyclic_image": cyclic_image,
            "same_category_swapped": shown_case["id"] != case["id"],
            "prompt_intrinsics_case_id": intrinsics_case["id"],
            "model_id": config["model_id"],
        }
        started = time.perf_counter()
        try:
            serialization_example = config.get("serialization_example")
            serialization_template_only = bool(
                config.get("serialization_template_only", False)
            )
            requested_category = config.get("category_overrides", {}).get(
                case["id"], case["category"]
            )
            prompt = build_prompt(
                requested_category, prompt_intrinsics, serialization_example,
                include_intrinsics, explicit_no_object,
                serialization_template_only,
            )
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
                "serialization_example": serialization_example,
                "serialization_template_only": serialization_template_only,
                "intrinsics_scale": intrinsics_scale,
                "include_intrinsics": include_intrinsics,
                "explicit_no_object": explicit_no_object,
                "white_frame": white_frame,
                "requested_category": requested_category,
                "prompt_intrinsics": prompt_intrinsics,
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
            shown_reference = shown_case.get("recorded_reference")
            if boxes and shown_reference is not None:
                result["shown_image_reference_center_error_m"] = math.dist(
                    boxes[0][:3], shown_reference[:3]
                )
            if boxes and serialization_example is not None:
                result["serialization_anchor_center_error_m"] = math.dist(
                    boxes[0][:3], serialization_example[:3]
                )
                result["serialization_anchor_full_mae"] = statistics.fmean(
                    abs(boxes[0][index] - serialization_example[index])
                    for index in range(9)
                )
            baseline = CALIBRATION_BASELINES.get(case["id"])
            if boxes and baseline is not None:
                expected = [
                    baseline[0] / intrinsics_scale,
                    baseline[1] / intrinsics_scale,
                    baseline[2],
                ]
                result["calibration_baseline_center"] = baseline
                result["calibration_expected_center"] = expected
                result["calibration_expected_center_error_m"] = math.dist(
                    boxes[0][:3], expected
                )
                result["calibration_unscaled_center_error_m"] = math.dist(
                    boxes[0][:3], baseline
                )
                result["calibration_expected_x_error_m"] = abs(
                    boxes[0][0] - expected[0]
                )
                result["calibration_unscaled_x_error_m"] = abs(
                    boxes[0][0] - baseline[0]
                )
                result["calibration_expected_y_error_m"] = abs(
                    boxes[0][1] - expected[1]
                )
                result["calibration_unscaled_y_error_m"] = abs(
                    boxes[0][1] - baseline[1]
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
    shown_referenced = [
        row for row in parsed if "shown_image_reference_center_error_m" in row
    ]
    swapped_referenced = [
        row for row in shown_referenced
        if row.get("same_category_swapped") and "recorded_center_error_m" in row
    ]
    anchored = [
        row for row in parsed if "serialization_anchor_center_error_m" in row
    ]
    calibrated = [
        row for row in parsed if "calibration_expected_center_error_m" in row
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
        "mean_shown_image_reference_center_error_m": statistics.fmean(
            row["shown_image_reference_center_error_m"] for row in shown_referenced
        ) if shown_referenced else None,
        "mean_swapped_original_reference_center_error_m": statistics.fmean(
            row["recorded_center_error_m"] for row in swapped_referenced
        ) if swapped_referenced else None,
        "mean_swapped_shown_reference_center_error_m": statistics.fmean(
            row["shown_image_reference_center_error_m"] for row in swapped_referenced
        ) if swapped_referenced else None,
        "mean_serialization_anchor_center_error_m": statistics.fmean(
            row["serialization_anchor_center_error_m"] for row in anchored
        ) if anchored else None,
        "mean_serialization_anchor_full_mae": statistics.fmean(
            row["serialization_anchor_full_mae"] for row in anchored
        ) if anchored else None,
        "mean_calibration_expected_center_error_m": statistics.fmean(
            row["calibration_expected_center_error_m"] for row in calibrated
        ) if calibrated else None,
        "mean_calibration_unscaled_center_error_m": statistics.fmean(
            row["calibration_unscaled_center_error_m"] for row in calibrated
        ) if calibrated else None,
        "mean_calibration_expected_x_error_m": statistics.fmean(
            row["calibration_expected_x_error_m"] for row in calibrated
        ) if calibrated else None,
        "mean_calibration_unscaled_x_error_m": statistics.fmean(
            row["calibration_unscaled_x_error_m"] for row in calibrated
        ) if calibrated else None,
        "mean_calibration_expected_y_error_m": statistics.fmean(
            row["calibration_expected_y_error_m"] for row in calibrated
        ) if calibrated else None,
        "mean_calibration_unscaled_y_error_m": statistics.fmean(
            row["calibration_unscaled_y_error_m"] for row in calibrated
        ) if calibrated else None,
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
