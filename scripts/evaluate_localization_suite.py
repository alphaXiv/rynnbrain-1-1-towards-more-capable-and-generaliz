#!/usr/bin/env python3
"""Distributed deterministic evaluation of official localization cookbooks."""

from __future__ import annotations

import copy
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
DATA = ROOT / "data" / "localization_cookbook"
ASSETS = ROOT / "cookbooks" / "assets"
RESULTS = Path("/tmp/rynnbrain-localization-results")
POINT_PATTERN = re.compile(
    r"\(\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*\)"
)
FRAME_PATTERN = re.compile(r"<frame\s+(\d+)>", re.IGNORECASE)


def parse_response(text: str, task: str) -> tuple[list[list[float]], int | None, bool]:
    tag_pattern = re.compile(
        rf"<{re.escape(task)}>\s*(.*?)\s*</{re.escape(task)}>",
        re.IGNORECASE | re.DOTALL,
    )
    match = tag_pattern.search(text)
    if not match:
        return [], None, False
    content = match.group(1)
    points = [[float(x), float(y)] for x, y in POINT_PATTERN.findall(content)]
    frame_match = FRAME_PATTERN.search(content)
    frame = int(frame_match.group(1)) if frame_match else None
    return points, frame, True


def point_count_valid(task: str, points: list[list[float]]) -> bool:
    if task == "object":
        return len(points) == 2
    if task == "affordance":
        return len(points) == 1
    if task == "trajectory":
        return 1 <= len(points) <= 10
    return bool(points)


def bbox_iou(first: list[list[float]], second: list[list[float]]) -> float:
    ax1, ay1 = first[0]
    ax2, ay2 = first[1]
    bx1, by1 = second[0]
    bx2, by2 = second[1]
    ax1, ax2 = sorted((ax1, ax2))
    ay1, ay2 = sorted((ay1, ay2))
    bx1, bx2 = sorted((bx1, bx2))
    by1, by2 = sorted((by1, by2))
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def point_chamfer(first: list[list[float]], second: list[list[float]]) -> float:
    def directed(source: list[list[float]], target: list[list[float]]) -> float:
        return statistics.fmean(
            min(math.dist(point, reference) for reference in target)
            for point in source
        )

    return (directed(first, second) + directed(second, first)) / 2


def build_content(case: dict[str, object]) -> list[dict[str, str]]:
    image_paths = [ASSETS / path for path in case["images"]]
    content: list[dict[str, str]] = []
    if len(image_paths) == 1:
        content.append({"type": "image", "image": str(image_paths[0])})
    else:
        for index, image_path in enumerate(image_paths):
            content.append({"type": "text", "text": f"<frame {index}>: "})
            content.append({"type": "image", "image": str(image_path)})
    prompt = f"{case['instruction']}\n{case['format_prompt']}"
    content.append({"type": "text", "text": prompt})
    return content


def horizontally_flipped_case(
    case: dict[str, object], rank: int
) -> dict[str, object]:
    transformed = copy.deepcopy(case)
    flipped_images = []
    for index, relative_path in enumerate(case["images"]):
        source_path = ASSETS / relative_path
        target_path = RESULTS / f"flip-rank-{rank}-image-{index}.png"
        with Image.open(source_path) as source_image:
            source_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(target_path)
        flipped_images.append(str(target_path))
    transformed["images"] = flipped_images
    transformed_points = [
        [1000 - point[0], point[1]] for point in case["recorded_points"]
    ]
    if case["task"] == "object":
        transformed_points.reverse()
    transformed["recorded_points"] = transformed_points
    transformed["source_recorded_points"] = case["recorded_points"]
    return transformed


def grayscale_case(case: dict[str, object], rank: int) -> dict[str, object]:
    transformed = copy.deepcopy(case)
    grayscale_images = []
    for index, relative_path in enumerate(case["images"]):
        source_path = ASSETS / relative_path
        target_path = RESULTS / f"grayscale-rank-{rank}-image-{index}.png"
        with Image.open(source_path) as source_image:
            source_image.convert("L").convert("RGB").save(target_path)
        grayscale_images.append(str(target_path))
    transformed["images"] = grayscale_images
    return transformed


def white_frame_case(case: dict[str, object], rank: int) -> dict[str, object]:
    transformed = copy.deepcopy(case)
    blank_images = []
    for index, relative_path in enumerate(case["images"]):
        source_path = ASSETS / relative_path
        target_path = RESULTS / f"white-rank-{rank}-image-{index}.png"
        with Image.open(source_path) as source_image:
            Image.new("RGB", source_image.size, "white").save(target_path)
        blank_images.append(str(target_path))
    transformed["images"] = blank_images
    return transformed


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
    horizontal_flip = bool(config.get("horizontal_flip", False))
    if horizontal_flip:
        case = horizontally_flipped_case(case, rank)
    grayscale = bool(config.get("grayscale", False))
    if grayscale:
        case = grayscale_case(case, rank)
    white_frame = bool(config.get("white_frame", False))
    if white_frame:
        case = white_frame_case(case, rank)
    result: dict[str, object] = {
        "rank": rank,
        "case_id": case["id"],
        "task": case["task"],
        "model_id": config["model_id"],
        "image_count": len(case["images"]),
        "horizontal_flip": horizontal_flip,
        "grayscale": grayscale,
        "white_frame": white_frame,
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
        content = build_content(case)
        conversation = [{"role": "user", "content": content}]
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
        result.update({
            "instruction": case["instruction"],
            "response": response,
            "points": points,
            "frame": frame,
            "tag_valid": tag_valid,
            "point_count_valid": count_valid,
            "coordinate_valid": coordinate_valid,
            "frame_valid": frame_valid,
            "format_valid": format_valid,
            "inference_seconds": time.perf_counter() - inference_started,
            "total_seconds": time.perf_counter() - started,
            "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        })
        reference_points = case["recorded_points"]
        result["recorded_points"] = reference_points
        if "source_recorded_points" in case:
            result["source_recorded_points"] = case["source_recorded_points"]
        if coordinate_valid:
            if case["task"] == "object":
                result["recorded_bbox_iou"] = bbox_iou(points, reference_points)
            else:
                chamfer = point_chamfer(points, reference_points)
                result["recorded_point_chamfer"] = chamfer
                result["recorded_point_chamfer_normalized"] = chamfer / 1000
        if "recorded_frame" in case:
            result["recorded_frame"] = case["recorded_frame"]
            result["recorded_frame_match"] = frame == case["recorded_frame"]
    except Exception as exc:
        result.update({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "total_seconds": time.perf_counter() - started,
        })

    (RESULTS / f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
    dist.barrier()

    if rank == 0:
        rows = [
            json.loads((RESULTS / f"rank-{index}.json").read_text())
            for index in range(world)
        ]
        for row in rows:
            print(
                "LOCALIZATION_CASE_RESULT " + json.dumps(row, sort_keys=True),
                flush=True,
            )
        successes = [row for row in rows if "error" not in row]
        formatted = [row for row in successes if row.get("format_valid")]
        coordinates = [row for row in successes if row.get("coordinate_valid")]
        frames = [row for row in successes if "recorded_frame_match" in row]
        bboxes = [row for row in coordinates if "recorded_bbox_iou" in row]
        point_cases = [
            row for row in coordinates if "recorded_point_chamfer" in row
        ]
        per_task = {}
        for task in ("object", "area", "affordance", "trajectory"):
            task_rows = [row for row in successes if row["task"] == task]
            per_task[task] = {
                "cases": len(task_rows),
                "format_valid_rate": (
                    sum(bool(row.get("format_valid")) for row in task_rows)
                    / len(task_rows)
                    if task_rows
                    else None
                ),
            }
        summary = {
            "model_id": config["model_id"],
            "cases": len(rows),
            "worker_successes": len(successes),
            "format_valid_rate": len(formatted) / len(rows),
            "coordinate_valid_rate": len(coordinates) / len(rows),
            "frame_valid_rate": sum(bool(row.get("frame_valid")) for row in successes)
            / len(rows),
            "recorded_frame_match_rate": (
                sum(bool(row["recorded_frame_match"]) for row in frames) / len(frames)
                if frames
                else None
            ),
            "mean_recorded_bbox_iou": (
                statistics.fmean(row["recorded_bbox_iou"] for row in bboxes)
                if bboxes
                else None
            ),
            "mean_recorded_point_chamfer": (
                statistics.fmean(
                    row["recorded_point_chamfer"] for row in point_cases
                )
                if point_cases
                else None
            ),
            "per_task": per_task,
            "mean_inference_seconds": (
                statistics.fmean(row["inference_seconds"] for row in successes)
                if successes
                else None
            ),
            "max_peak_gpu_gib": max(
                (row["peak_gpu_gib"] for row in successes), default=None
            ),
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
        }
        print(
            "LOCALIZATION_EVAL_SUMMARY " + json.dumps(summary, sort_keys=True),
            flush=True,
        )
        if len(successes) != len(rows):
            raise SystemExit(2)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
