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
from PIL import Image, ImageFilter
from transformers import AutoModelForImageTextToText, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "three_d_grounding"
IMAGES = ROOT / "cookbooks" / "assets" / "3d_grounding" / "images"
TAG_PATTERN = re.compile(
    r"<3D\s+Grounding>\s*(.*?)\s*</3D\s+Grounding>", re.IGNORECASE | re.DOTALL
)
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
CALIBRATION_BASELINES = {
    "sunrgbd_chair": [-0.16, -0.01, 1.37],
    "sunrgbd_bed": [-0.12, -0.06, 3.22],
    "sunrgbd_table": [0.31, -0.19, 1.97],
    "office_chairs": [-0.72, -0.52, 6.44],
    "lounge_sofa": [-1.14, -0.36, 3.61],
    "lounge_table": [0.40, 0.13, 2.20],
    "manipulation_bottle": [0.05, -0.06, 1.17],
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
    return f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}"


def build_prompt(
    category: str,
    intrinsics: list[float],
    include_intrinsics: bool = True,
    x_direction: str = "right",
    y_direction: str = "downward",
    coordinate_units: str = "meters",
    center_representation: str = "camera_xyz",
    serialization_example: list[float] | None = None,
    serialization_template_only: bool = False,
) -> str:
    intrinsics_block = (
        f"The camera intrinsics are:\n{format_intrinsics(intrinsics)}\n\n"
        if include_intrinsics
        else ""
    )
    if center_representation == "pixel_uvz":
        center_fields = "u, v, z"
        center_definition = "u, v: pixel coordinates of the 3D box center; z: depth in meters"
        units_constraint = "Use pixels for u, v and meters for z, x_size, y_size, z_size"
    elif center_representation == "normalized_uvz":
        center_fields = "u_1000, v_1000, z"
        center_definition = (
            "u_1000, v_1000: image coordinates of the 3D box center normalized "
            "to [0, 1000]; z: depth in meters"
        )
        units_constraint = (
            "Use [0, 1000] normalized image coordinates for u_1000, v_1000 "
            "and meters for z, x_size, y_size, z_size"
        )
    else:
        center_fields = "cx, cy, cz"
        center_definition = (
            "cx, cy, cz: 3D coordinates of the box center in the camera "
            f"coordinate system, in {coordinate_units}"
        )
        units_constraint = (
            f"Use {coordinate_units} for cx, cy, cz, x_size, y_size, z_size"
        )
    example = ""
    if serialization_template_only:
        example = (
            "\nFormat template (field names are placeholders, not values):\n"
            f"<3D Grounding> {center_fields}, x_size, y_size, z_size, "
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

{intrinsics_block}Predict 3D bounding boxes in the camera coordinate system, where:
- x points to the {x_direction}
- y points {y_direction}
- z points forward

For each object, return:
<3D Grounding> {center_fields}, x_size, y_size, z_size, pitch, yaw, roll </3D Grounding>

Definitions:
- {center_definition}
- x_size, y_size, z_size: box dimensions in the box local coordinate system, in {coordinate_units}
- pitch, yaw, roll: normalized rotations in the range [-1, 1], corresponding to [-180, 180] degrees

Constraints:
- x_size >= z_size
- {units_constraint}
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
    box: list[float],
    intrinsics: list[float],
    image_size: tuple[int, int],
    x_axis_sign: float = 1.0,
    y_axis_sign: float = 1.0,
) -> bool:
    if box[2] <= 0:
        return False
    fx, fy, cx, cy = intrinsics
    u = fx * x_axis_sign * box[0] / box[2] + cx
    v = fy * y_axis_sign * box[1] / box[2] + cy
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
        fixed_image_case_id = config.get("fixed_image_case_id")
        shown_case = (
            cases_by_id[fixed_image_case_id]
            if fixed_image_case_id is not None
            else case
        )
        image_path = IMAGES / shown_case["image"]
        focal_scale_sweep = config.get("focal_scale_sweep")
        intrinsics_case = shown_case if focal_scale_sweep is not None else case
        focal_scale = (
            float(focal_scale_sweep[rank])
            if focal_scale_sweep is not None
            else 1.0
        )
        prompt_intrinsics = list(intrinsics_case["intrinsics"])
        intrinsics_scale = float(config.get("intrinsics_scale", 1.0))
        include_intrinsics = bool(config.get("include_intrinsics", True))
        white_frame = bool(config.get("white_frame", False))
        blur_radius = float(config.get("blur_radius", 0.0))
        grayscale_frame = bool(config.get("grayscale_frame", False))
        image_scale = float(config.get("image_scale", 1.0))
        crop_fraction = float(config.get("crop_fraction", 1.0))
        x_axis_sign = float(config.get("x_axis_sign", 1.0))
        y_axis_sign = float(config.get("y_axis_sign", 1.0))
        coordinate_scale = float(config.get("coordinate_scale", 1.0))
        coordinate_units = str(config.get("coordinate_units", "meters"))
        center_representation = str(
            config.get("center_representation", "camera_xyz")
        )
        prompt_intrinsics[0] *= intrinsics_scale * focal_scale
        prompt_intrinsics[1] *= intrinsics_scale * focal_scale
        prompt_intrinsics = [value * image_scale for value in prompt_intrinsics]
        model_image_path = image_path
        with Image.open(image_path) as source_image:
            image_size = source_image.size
            if crop_fraction < 1.0:
                width, height = source_image.size
                left = round((1.0 - crop_fraction) * width / 2)
                top = round((1.0 - crop_fraction) * height / 2)
                right = width - left
                bottom = height - top
                image_size = (right - left, bottom - top)
                prompt_intrinsics[2] -= left
                prompt_intrinsics[3] -= top
                model_image_path = Path(f"/tmp/cropped-frame-rank-{rank}.png")
                source_image.convert("RGB").crop(
                    (left, top, right, bottom)
                ).save(model_image_path)
            elif image_scale != 1.0:
                image_size = tuple(
                    max(1, round(dimension * image_scale))
                    for dimension in source_image.size
                )
                model_image_path = Path(f"/tmp/scaled-frame-rank-{rank}.png")
                source_image.convert("RGB").resize(
                    image_size, Image.Resampling.LANCZOS
                ).save(model_image_path)
            elif white_frame:
                model_image_path = Path(f"/tmp/white-frame-rank-{rank}.png")
                Image.new("RGB", image_size, "white").save(model_image_path)
            elif grayscale_frame:
                model_image_path = Path(f"/tmp/grayscale-frame-rank-{rank}.png")
                source_image.convert("L").convert("RGB").save(model_image_path)
            elif blur_radius > 0:
                model_image_path = Path(f"/tmp/blurred-frame-rank-{rank}.png")
                source_image.convert("RGB").filter(
                    ImageFilter.GaussianBlur(radius=blur_radius)
                ).save(model_image_path)
        result: dict[str, object] = {
            "rank": rank,
            "case_id": case["id"],
            "category": case["category"],
            "shown_case_id": shown_case["id"],
            "shown_category": shown_case["category"],
            "prompt_intrinsics_case_id": intrinsics_case["id"],
            "focal_scale": focal_scale,
            "model_id": config["model_id"],
        }
        started = time.perf_counter()
        try:
            requested_category = config.get("fixed_requested_category") or (
                config.get("category_overrides", {}).get(
                    case["id"], case["category"]
                )
            )
            serialization_example = config.get("serialization_example")
            serialization_template_only = bool(
                config.get("serialization_template_only", False)
            )
            prompt = build_prompt(
                requested_category,
                prompt_intrinsics,
                include_intrinsics,
                "left" if x_axis_sign < 0 else "right",
                "upward" if y_axis_sign < 0 else "downward",
                coordinate_units,
                center_representation,
                serialization_example,
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
            camera_boxes = boxes
            if center_representation in {"pixel_uvz", "normalized_uvz"}:
                fx, fy, cx, cy = prompt_intrinsics
                width, height = image_size
                camera_boxes = [
                    [
                        (
                            (box[0] * width / 1000 if center_representation == "normalized_uvz" else box[0])
                            - cx
                        ) * box[2] / fx,
                        (
                            (box[1] * height / 1000 if center_representation == "normalized_uvz" else box[1])
                            - cy
                        ) * box[2] / fy,
                        box[2],
                        *box[3:],
                    ]
                    for box in boxes
                ]
            physical = bool(
                camera_boxes and all(box_is_physical(box) for box in camera_boxes)
            )
            projected = bool(
                boxes
                and any(
                    center_projects_inside(
                        box,
                        prompt_intrinsics,
                        image_size,
                        x_axis_sign,
                        y_axis_sign,
                    )
                    for box in camera_boxes
                )
            )
            result.update({
                "prompt": prompt,
                "response": response,
                "boxes": boxes,
                "camera_boxes": camera_boxes,
                "intrinsics_scale": intrinsics_scale,
                "include_intrinsics": include_intrinsics,
                "white_frame": white_frame,
                "blur_radius": blur_radius,
                "grayscale_frame": grayscale_frame,
                "image_scale": image_scale,
                "crop_fraction": crop_fraction,
                "x_axis_sign": x_axis_sign,
                "y_axis_sign": y_axis_sign,
                "coordinate_scale": coordinate_scale,
                "coordinate_units": coordinate_units,
                "center_representation": center_representation,
                "serialization_example": serialization_example,
                "serialization_template_only": serialization_template_only,
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
            if camera_boxes and reference is not None and coordinate_scale == 1.0:
                result["recorded_reference"] = reference
                result["recorded_center_error_m"] = math.dist(
                    camera_boxes[0][:3], reference[:3]
                )
                result["recorded_dimension_error_m"] = math.dist(
                    camera_boxes[0][3:6], reference[3:6]
                )
                result["recorded_angle_mae"] = statistics.fmean(
                    abs(camera_boxes[0][index] - reference[index])
                    for index in range(6, 9)
                )
            shown_reference = shown_case.get("recorded_reference")
            if camera_boxes and shown_reference is not None and coordinate_scale == 1.0:
                result["shown_image_reference_center_error_m"] = math.dist(
                    camera_boxes[0][:3], shown_reference[:3]
                )
            if camera_boxes and serialization_example is not None:
                result["serialization_anchor_center_error_m"] = math.dist(
                    camera_boxes[0][:3], serialization_example[:3]
                )
                result["serialization_anchor_full_mae"] = statistics.fmean(
                    abs(camera_boxes[0][index] - serialization_example[index])
                    for index in range(9)
                )
            baseline = CALIBRATION_BASELINES.get(case["id"])
            if camera_boxes and baseline is not None:
                expected = [
                    coordinate_scale * x_axis_sign * baseline[0] / intrinsics_scale,
                    coordinate_scale * y_axis_sign * baseline[1] / intrinsics_scale,
                    coordinate_scale * baseline[2],
                ]
                result["calibration_baseline_center"] = baseline
                result["calibration_expected_center"] = expected
                result["calibration_expected_center_error_m"] = math.dist(
                    camera_boxes[0][:3], expected
                )
                result["calibration_unscaled_center_error_m"] = math.dist(
                    camera_boxes[0][:3], baseline
                )
                result["calibration_expected_x_error_m"] = abs(
                    camera_boxes[0][0] - expected[0]
                )
                result["calibration_unscaled_x_error_m"] = abs(
                    camera_boxes[0][0] - baseline[0]
                )
                result["calibration_expected_y_error_m"] = abs(
                    camera_boxes[0][1] - expected[1]
                )
                result["calibration_unscaled_y_error_m"] = abs(
                    camera_boxes[0][1] - baseline[1]
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
