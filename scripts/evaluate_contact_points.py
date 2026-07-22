#!/usr/bin/env python3
"""Distributed deterministic evaluation on the eight official contact cases."""

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
from transformers import AutoModelForImageTextToText, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "contact_point_prediction"
RESULTS = Path("/tmp/rynnbrain-contact-results")
PATTERN = re.compile(
    r"<grasp\s+pose>\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*</grasp\s+pose>",
    re.IGNORECASE,
)


def parse_pose(text: str) -> tuple[float, float, float] | None:
    match = PATTERN.search(text)
    return tuple(map(float, match.groups())) if match else None


def circular_gripper_error(a: float, b: float) -> float:
    """Smallest angular difference under a parallel-jaw gripper's 180° symmetry."""
    raw = abs(a - b) % 180.0
    return min(raw, 180.0 - raw)


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    config = json.loads((ROOT / "config.json").read_text())
    cases = json.loads((DATA / "test_cases.json").read_text())
    references = {
        row["id"]: row for row in json.loads((DATA / "recorded_results.json").read_text())
    }
    if world != len(cases):
        raise RuntimeError(f"expected {len(cases)} ranks, received {world}")

    model_dir = Path("/tmp/model-snapshot")
    if rank == 0:
        RESULTS.mkdir(parents=True, exist_ok=True)
        print(f"Downloading immutable snapshot for {config['model_id']}", flush=True)
        snapshot_download(
            repo_id=config["model_id"],
            local_dir=model_dir,
            max_workers=16,
        )
    dist.barrier()

    case = cases[rank]
    result: dict[str, object] = {
        "rank": rank,
        "case_id": case["id"],
        "group": case["group"],
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
        evaluated_query = case["query"] + config.get("query_suffix", "")
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(DATA / case["image"])},
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
        pose = parse_pose(response)
        ref_pose = parse_pose(references[case["id"]]["pred"])
        valid = pose is not None and all(math.isfinite(x) for x in pose)
        in_bounds = bool(valid and 0 <= pose[0] <= 1000 and 0 <= pose[1] <= 1000)
        result.update({
            "query": evaluated_query,
            "response": response,
            "pose": pose,
            "reference_pose": ref_pose,
            "format_valid": valid,
            "coordinate_valid": in_bounds,
            "inference_seconds": time.perf_counter() - inference_started,
            "total_seconds": time.perf_counter() - started,
            "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        })
        if valid and ref_pose is not None:
            result["reference_position_error"] = math.dist(pose[:2], ref_pose[:2])
            result["reference_angle_error_deg"] = circular_gripper_error(pose[2], ref_pose[2])
    except Exception as exc:  # keep every rank rendezvousing so rank 0 can report evidence
        result.update({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "total_seconds": time.perf_counter() - started,
        })

    (RESULTS / f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
    dist.barrier()

    if rank == 0:
        rows = [json.loads((RESULTS / f"rank-{i}.json").read_text()) for i in range(world)]
        for row in rows:
            print("CASE_RESULT " + json.dumps(row, sort_keys=True), flush=True)
        successes = [row for row in rows if "error" not in row]
        parsed = [row for row in successes if row.get("format_valid")]
        bounded = [row for row in successes if row.get("coordinate_valid")]
        position_errors = [row["reference_position_error"] for row in parsed]
        angle_errors = [row["reference_angle_error_deg"] for row in parsed]
        summary = {
            "model_id": config["model_id"],
            "cases": len(rows),
            "worker_successes": len(successes),
            "format_valid_rate": len(parsed) / len(rows),
            "coordinate_valid_rate": len(bounded) / len(rows),
            "mean_reference_position_error": statistics.fmean(position_errors) if position_errors else None,
            "mean_reference_angle_error_deg": statistics.fmean(angle_errors) if angle_errors else None,
            "mean_inference_seconds": statistics.fmean(
                row["inference_seconds"] for row in successes
            ) if successes else None,
            "max_peak_gpu_gib": max((row["peak_gpu_gib"] for row in successes), default=None),
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
        }
        print("EVAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
        if len(successes) != len(rows):
            raise SystemExit(2)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
