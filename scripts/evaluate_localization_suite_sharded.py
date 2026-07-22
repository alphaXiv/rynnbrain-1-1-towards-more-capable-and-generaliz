#!/usr/bin/env python3
"""Eight-GPU balanced-dispatch evaluation of official localization cookbooks."""

from __future__ import annotations

import copy
import json
import statistics
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluate_localization_suite import (
    ASSETS,
    DATA,
    ROOT,
    bbox_iou,
    build_content,
    parse_response,
    point_chamfer,
    point_count_valid,
)


def apply_inset_letterbox(case: dict[str, object], scale: float) -> None:
    """Shrink each frame around its center and update normalized anchors."""
    output_dir = Path("/tmp/inset-letterbox") / str(case["id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    transformed_images: list[str] = []
    for index, relative_path in enumerate(case["images"]):
        with Image.open(ASSETS / relative_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        inset_width = round(width * scale)
        inset_height = round(height * scale)
        canvas = Image.new("RGB", (width, height), "white")
        resized = image.resize((inset_width, inset_height), Image.Resampling.BICUBIC)
        canvas.paste(resized, ((width - inset_width) // 2, (height - inset_height) // 2))
        output_path = output_dir / f"frame-{index:02d}.png"
        canvas.save(output_path)
        transformed_images.append(str(output_path))
    case["images"] = transformed_images
    offset = 500.0 * (1.0 - scale)
    case["recorded_points"] = [
        [offset + scale * x, offset + scale * y]
        for x, y in case["recorded_points"]
    ]


def apply_clockwise_rotation(case: dict[str, object]) -> None:
    """Rotate every frame 90 degrees clockwise and update normalized anchors."""
    output_dir = Path("/tmp/clockwise-rotation") / str(case["id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    transformed_images: list[str] = []
    for index, relative_path in enumerate(case["images"]):
        with Image.open(ASSETS / relative_path) as source:
            image = source.convert("RGB")
        rotated = image.transpose(Image.Transpose.ROTATE_270)
        output_path = output_dir / f"frame-{index:02d}.png"
        rotated.save(output_path)
        transformed_images.append(str(output_path))
    case["images"] = transformed_images
    case["recorded_points"] = [
        [1000.0 - y, x] for x, y in case["recorded_points"]
    ]


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
    reverse_video_frames = bool(config.get("reverse_video_frames", False))
    inset_letterbox_scale = config.get("inset_letterbox_scale")
    clockwise_rotation = bool(config.get("clockwise_rotation_90", False))
    for source_case in cases:
        case = copy.deepcopy(source_case)
        source_recorded_frame = case.get("recorded_frame")
        source_recorded_points = copy.deepcopy(case["recorded_points"])
        if reverse_video_frames and len(case["images"]) > 1:
            case["images"].reverse()
            if source_recorded_frame is not None:
                case["recorded_frame"] = len(case["images"]) - 1 - source_recorded_frame
        if inset_letterbox_scale is not None:
            apply_inset_letterbox(case, float(inset_letterbox_scale))
        if clockwise_rotation:
            apply_clockwise_rotation(case)
        conversation = [{"role": "user", "content": build_content(case)}]
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
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(config["max_new_tokens"]),
                do_sample=False,
            )
        for index in range(gpu_count):
            torch.cuda.synchronize(index)
        elapsed = time.perf_counter() - started
        peak_gib = max(
            torch.cuda.max_memory_allocated(index) / (1024**3)
            for index in range(gpu_count)
        )
        output_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        response = processor.decode(output_ids[0], skip_special_tokens=True)
        points, frame, tag_valid = parse_response(response, case["task"])
        count_valid = point_count_valid(case["task"], points)
        coordinate_valid = bool(
            count_valid and all(0 <= value <= 1000 for point in points for value in point)
        )
        needs_frame = len(case["images"]) > 1
        frame_valid = bool(
            (not needs_frame and frame is None)
            or (needs_frame and frame is not None and 0 <= frame < len(case["images"]))
        )
        format_valid = bool(tag_valid and count_valid and frame_valid)
        result: dict[str, object] = {
            "case_id": case["id"],
            "task": case["task"],
            "model_id": config["model_id"],
            "image_count": len(case["images"]),
            "reverse_video_frames": reverse_video_frames,
            "inset_letterbox_scale": inset_letterbox_scale,
            "clockwise_rotation_90": clockwise_rotation,
            "source_recorded_frame": source_recorded_frame,
            "source_recorded_points": source_recorded_points,
            "instruction": case["instruction"],
            "response": response,
            "points": points,
            "frame": frame,
            "tag_valid": tag_valid,
            "point_count_valid": count_valid,
            "coordinate_valid": coordinate_valid,
            "frame_valid": frame_valid,
            "format_valid": format_valid,
            "inference_seconds": elapsed,
            "peak_gpu_gib": peak_gib,
            "recorded_points": case["recorded_points"],
        }
        if coordinate_valid:
            if case["task"] == "object":
                result["recorded_bbox_iou"] = bbox_iou(
                    points, case["recorded_points"]
                )
            else:
                chamfer = point_chamfer(points, case["recorded_points"])
                result["recorded_point_chamfer"] = chamfer
                result["recorded_point_chamfer_normalized"] = chamfer / 1000
        if "recorded_frame" in case:
            result["recorded_frame"] = case["recorded_frame"]
            result["recorded_frame_match"] = frame == case["recorded_frame"]
        rows.append(result)
        print(
            "LOCALIZATION_CASE_RESULT " + json.dumps(result, sort_keys=True),
            flush=True,
        )

    formatted = [row for row in rows if row["format_valid"]]
    coordinates = [row for row in rows if row["coordinate_valid"]]
    frames = [row for row in rows if "recorded_frame_match" in row]
    bboxes = [row for row in coordinates if "recorded_bbox_iou" in row]
    point_cases = [
        row for row in coordinates if "recorded_point_chamfer" in row
    ]
    per_task = {}
    for task in ("object", "area", "affordance", "trajectory"):
        task_rows = [row for row in rows if row["task"] == task]
        per_task[task] = {
            "cases": len(task_rows),
            "format_valid_rate": sum(
                bool(row["format_valid"]) for row in task_rows
            ) / len(task_rows),
        }
    summary = {
        "model_id": config["model_id"],
        "cases": len(rows),
        "worker_successes": 1,
        "sharded_gpu_count": gpu_count,
        "format_valid_rate": len(formatted) / len(rows),
        "coordinate_valid_rate": len(coordinates) / len(rows),
        "frame_valid_rate": sum(bool(row["frame_valid"]) for row in rows)
        / len(rows),
        "recorded_frame_match_rate": sum(
            bool(row["recorded_frame_match"]) for row in frames
        ) / len(frames),
        "mean_recorded_bbox_iou": (
            statistics.fmean(row["recorded_bbox_iou"] for row in bboxes)
            if bboxes
            else None
        ),
        "mean_recorded_point_chamfer": (
            statistics.fmean(row["recorded_point_chamfer"] for row in point_cases)
            if point_cases
            else None
        ),
        "per_task": per_task,
        "mean_inference_seconds": statistics.fmean(
            row["inference_seconds"] for row in rows
        ),
        "max_peak_gpu_gib": max(row["peak_gpu_gib"] for row in rows),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    print(
        "LOCALIZATION_EVAL_SUMMARY " + json.dumps(summary, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
