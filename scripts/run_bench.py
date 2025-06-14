#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_bench.py – Universal PyTorch latency probe using model *block* modules.
Python 3.11
"""

from __future__ import annotations

import argparse
import csv
import importlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

import torch

# ---------------------------------------------------------------------------
# 모델 key 목록 (block 파일 이름 prefix)
# ---------------------------------------------------------------------------
VISION_KEYS = ["conv", "resnet", "mobilenet", "vit"]
GNN_KEYS    = ["gcn", "graphsage", "gat", "gatv2"]
LLM_KEYS    = ["bert", "gpt2", "llama", "deepseek"]
ALL_MODELS  = VISION_KEYS + GNN_KEYS + LLM_KEYS

# ---------------------------------------------------------------------------
# Model Loader – 모든 로직을 각 block 내부로 위임
# ---------------------------------------------------------------------------
def load_model(name: str):
    """
    Import `models/<name>_block.py` and return (model, dummy_input).

    Each block **must** implement:
        get_model()        -> torch.nn.Module (eval mode는 block 쪽 책임)
        get_dummy_input()  -> Tensor | tuple[Tensor,...] | shape-tuple[int,...]
    """
    name = name.lower()
    mod = importlib.import_module(f"models.{name}_block")
    model = mod.get_model()
    dummy = mod.get_dummy_input()
    return model, dummy


# ---------------------------------------------------------------------------
# Forward timer
# ---------------------------------------------------------------------------
def time_forward(model: torch.nn.Module, dummy_input: Any, device: str = "cuda") -> float:
    model = model.to(device)

    # ------------------------- input to device -----------------------------
    if isinstance(dummy_input, tuple):
        # ➊ tuple of tensors → Vision(GNN/LLM) positional call
        if all(isinstance(t, torch.Tensor) for t in dummy_input):
            tensors = [t.to(device) for t in dummy_input]
            fn = lambda: model(*tensors)
        # ➋ tuple of ints → shape for random tensor (Conv / vision models)
        else:
            x = torch.randn(*dummy_input, device=device)
            fn = lambda: model(x)
    else:
        # Single tensor → just call
        fn = lambda: model(dummy_input.to(device))

    # --------------------------- warm-up -----------------------------------
    for _ in range(5):
        with torch.no_grad():
            fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # --------------------------- timing ------------------------------------
    start = time.perf_counter()
    with torch.no_grad():
        fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000.0  # ms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="*", help=f"Models: {'|'.join(ALL_MODELS)}")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--csv", action="store_true", help="Save results to results/latency_*.csv")
    args = p.parse_args()

    models_to_run = args.model if args.model else ALL_MODELS
    results = []
    timestamp = datetime.now().isoformat(timespec="seconds")

    for model_name in models_to_run:
        try:
            mdl, dummy = load_model(model_name)
            ms = time_forward(mdl, dummy, args.device)
            print(f"{model_name:10s} | {args.device} | {ms:10.6f} ms")
            results.append((timestamp, model_name, args.device, f"{ms:.6f}"))
        except Exception as e:
            print(f"[ERROR] {model_name}: {e}")
            results.append((timestamp, model_name, args.device, str(e)))

    # --------------------------- CSV dump -----------------------------------
    if args.csv:
        tag = "all" if not args.model else "_".join(args.model)
        out_path = Path(f"results/latency_{tag}_{args.device}.csv")
        out_path.parent.mkdir(exist_ok=True)

        # avoid overwrite
        idx, candidate = 1, out_path
        while candidate.exists():
            candidate = out_path.with_stem(f"{out_path.stem}_{idx}")
            idx += 1

        with candidate.open("a", newline="") as f:
            csv.writer(f).writerows(results)
        print(f"[✓] Results saved to {candidate}")


if __name__ == "__main__":
    main()

