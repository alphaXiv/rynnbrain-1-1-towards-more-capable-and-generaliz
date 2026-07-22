#!/usr/bin/env python3
"""Eight-GPU balanced layer-dispatch evaluation of every official contact case."""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "contact_point_prediction"
PATTERN = re.compile(
    r"<grasp\s+pose>\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*</grasp\s+pose>",
    re.IGNORECASE,
)


def parse_pose(text: str) -> tuple[float, float, float] | None:
    match = PATTERN.search(text)
    return tuple(map(float, match.groups())) if match else None


def circular_gripper_error(a: float, b: float) -> float:
    raw = abs(a - b) % 180.0
    return min(raw, 180.0 - raw)


def prepare_image_and_reference(
    case: dict[str, object],
    reference: tuple[float, float, float] | None,
    transform: str | None,
) -> tuple[Path, tuple[float, float, float] | None]:
    image_path = DATA / str(case["image"])
    if transform is None:
        return image_path, reference
    if transform != "vertical_flip":
        raise ValueError(f"unsupported image_transform={transform!r}")

    transformed_dir = Path("/tmp/rynnbrain-transformed-inputs")
    transformed_dir.mkdir(parents=True, exist_ok=True)
    transformed_path = transformed_dir / f"{case['id']}.png"
    with Image.open(image_path) as image:
        image.convert("RGB").transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(
            transformed_path
        )
    if reference is not None:
        x, y, theta = reference
        reference = (x, 1000.0 - y, (-theta) % 180.0)
    return transformed_path, reference


def main() -> None:
    gpu_count = torch.cuda.device_count()
    if gpu_count != 8:
        raise RuntimeError(f"expected 8 visible GPUs, received {gpu_count}")
    input_device = torch.device("cuda:0")

    config = json.loads((ROOT / "config.json").read_text())
    cases = json.loads((DATA / "test_cases.json").read_text())
    references = {
        row["id"]: row for row in json.loads((DATA / "recorded_results.json").read_text())
    }
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
    for case in cases:
        evaluated_query = case["query"] + config.get("query_suffix", "")
        ref_pose = parse_pose(references[case["id"]]["pred"])
        image_path, ref_pose = prepare_image_and_reference(
            case, ref_pose, config.get("image_transform")
        )
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": evaluated_query},
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
        pose = parse_pose(response)
        valid = pose is not None and all(math.isfinite(x) for x in pose)
        in_bounds = bool(valid and 0 <= pose[0] <= 1000 and 0 <= pose[1] <= 1000)
        result: dict[str, object] = {
            "case_id": case["id"],
            "group": case["group"],
            "model_id": config["model_id"],
            "query": evaluated_query,
            "image_transform": config.get("image_transform"),
            "response": response,
            "pose": pose,
            "reference_pose": ref_pose,
            "format_valid": valid,
            "coordinate_valid": in_bounds,
            "inference_seconds": elapsed,
            "peak_gpu_gib": peak_gib,
        }
        if valid and ref_pose is not None:
            result["reference_position_error"] = math.dist(pose[:2], ref_pose[:2])
            result["reference_angle_error_deg"] = circular_gripper_error(
                pose[2], ref_pose[2]
            )
        rows.append(result)
        print("CASE_RESULT " + json.dumps(result, sort_keys=True), flush=True)

    parsed = [row for row in rows if row["format_valid"]]
    bounded = [row for row in rows if row["coordinate_valid"]]
    summary = {
        "model_id": config["model_id"],
        "cases": len(rows),
        "worker_successes": 1,
        "sharded_gpu_count": gpu_count,
        "format_valid_rate": len(parsed) / len(rows),
        "coordinate_valid_rate": len(bounded) / len(rows),
        "mean_reference_position_error": statistics.fmean(
            row["reference_position_error"] for row in parsed
        ) if parsed else None,
        "mean_reference_angle_error_deg": statistics.fmean(
            row["reference_angle_error_deg"] for row in parsed
        ) if parsed else None,
        "mean_inference_seconds": statistics.fmean(
            row["inference_seconds"] for row in rows
        ),
        "max_peak_gpu_gib": max(row["peak_gpu_gib"] for row in rows),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    print("EVAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
