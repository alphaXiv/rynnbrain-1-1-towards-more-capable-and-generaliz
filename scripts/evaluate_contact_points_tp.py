#!/usr/bin/env python3
"""Eight-way tensor-parallel evaluation of every official contact case."""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "contact_point_prediction"
PATTERN = re.compile(
    r"<grasp\s+pose>\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*</grasp\s+pose>",
    re.IGNORECASE,
)
TP_PLAN = {
    "model.language_model.layers.*.self_attn.q_proj": "colwise",
    "model.language_model.layers.*.self_attn.o_proj": "rowwise",
    "model.language_model.layers.*.mlp.experts.gate_up_proj": "packed_colwise",
    "model.language_model.layers.*.mlp.experts.down_proj": "rowwise",
    "model.language_model.layers.*.mlp.shared_expert.gate_proj": "colwise",
    "model.language_model.layers.*.mlp.shared_expert.up_proj": "colwise",
    "model.language_model.layers.*.mlp.shared_expert.down_proj": "rowwise",
}


def parse_pose(text: str) -> tuple[float, float, float] | None:
    match = PATTERN.search(text)
    return tuple(map(float, match.groups())) if match else None


def circular_gripper_error(a: float, b: float) -> float:
    raw = abs(a - b) % 180.0
    return min(raw, 180.0 - raw)


def distributed_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 8:
        raise RuntimeError(f"expected 8 tensor-parallel ranks, received {world}")

    config = json.loads((ROOT / "config.json").read_text())
    cases = json.loads((DATA / "test_cases.json").read_text())
    references = {
        row["id"]: row for row in json.loads((DATA / "recorded_results.json").read_text())
    }
    model_dir = Path("/tmp/model-snapshot")
    if rank == 0:
        print(f"Downloading immutable snapshot for {config['model_id']}", flush=True)
        snapshot_download(repo_id=config["model_id"], local_dir=model_dir, max_workers=32)
    dist.barrier()

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        tp_plan=TP_PLAN,
    ).eval()
    if rank == 0:
        print(
            "TP_MODEL_READY "
            + json.dumps({
                "model_id": config["model_id"],
                "world_size": world,
                "tp_plan": getattr(model, "_tp_plan", None),
            }, sort_keys=True, default=str),
            flush=True,
        )

    rows: list[dict[str, object]] = []
    for case in cases:
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
        torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(config["max_new_tokens"]),
                do_sample=False,
            )
        torch.cuda.synchronize(device)
        elapsed = distributed_max(time.perf_counter() - started, device)
        peak_gib = distributed_max(
            torch.cuda.max_memory_allocated(device) / (1024**3), device
        )
        output_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        response = processor.decode(output_ids[0], skip_special_tokens=True)
        if rank == 0:
            pose = parse_pose(response)
            ref_pose = parse_pose(references[case["id"]]["pred"])
            valid = pose is not None and all(math.isfinite(x) for x in pose)
            in_bounds = bool(valid and 0 <= pose[0] <= 1000 and 0 <= pose[1] <= 1000)
            result: dict[str, object] = {
                "case_id": case["id"],
                "group": case["group"],
                "model_id": config["model_id"],
                "query": evaluated_query,
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
        dist.barrier()

    if rank == 0:
        parsed = [row for row in rows if row["format_valid"]]
        bounded = [row for row in rows if row["coordinate_valid"]]
        summary = {
            "model_id": config["model_id"],
            "cases": len(rows),
            "worker_successes": world,
            "tensor_parallel_world_size": world,
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
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
        }
        print("EVAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
