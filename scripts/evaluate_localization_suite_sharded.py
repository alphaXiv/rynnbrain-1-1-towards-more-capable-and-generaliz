#!/usr/bin/env python3
"""Eight-GPU balanced-dispatch evaluation of official localization cookbooks."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluate_localization_suite import (
    DATA,
    RESULTS,
    ROOT,
    bbox_iou,
    build_content,
    parse_response,
    point_chamfer,
    point_count_valid,
    rotated_180_case,
)


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
    RESULTS.mkdir(parents=True, exist_ok=True)
    for rank, source_case in enumerate(cases):
        rotation_180 = bool(config.get("rotation_180", False))
        case = rotated_180_case(source_case, rank) if rotation_180 else source_case
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
            "rotation_180": rotation_180,
            "source_recorded_points": case.get("source_recorded_points"),
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
