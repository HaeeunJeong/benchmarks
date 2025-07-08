#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_bench.py – Universal PyTorch latency probe using model *block* modules.
Python 3.11
"""

from __future__ import annotations

import argparse, csv, importlib, time, sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add root to path

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

    각 block **must** implement:
        get_model()        -> torch.nn.Module (eval mode는 block 쪽 책임)
        get_dummy_input()  -> Tensor | tuple[Tensor,...] | shape-tuple[int,...]
    """
    mod = importlib.import_module(f"models.{name.lower()}_block")
    return mod.get_model(), mod.get_dummy_input()


# ---------------------------------------------------------------------------
# Forward timer
# ---------------------------------------------------------------------------
def time_forward(model: torch.nn.Module, dummy_input: Any, device: str = "cuda") -> float:
    """
    Return forward-pass latency in **milliseconds** (ms).

    All memory 이동, 컴파일, warm-up 오버헤드는 타이머 밖에서 제거.
    """
    device_obj = torch.device(device)
    model = model.to(device_obj).eval()

    # ------------------------- 입력을 device로 이동 --------------------------
    if isinstance(dummy_input, torch.Tensor):
        inp = dummy_input.to(device_obj)
        call = lambda: model(inp)

    elif isinstance(dummy_input, tuple):
        # ➊ tuple of Tensors: Vision/GNN/LLM positional 입력
        if all(isinstance(t, torch.Tensor) for t in dummy_input):
            tensors = [t.to(device_obj) for t in dummy_input]
            call = lambda: model(*tensors)
        # ➋ tuple of ints: shape 지정 → 랜덤 Tensor 생성
        else:
            x = torch.randn(*dummy_input, device=device_obj)
            call = lambda: model(x)

    else:
        raise TypeError("dummy_input must be Tensor or Tuple")

    # --------------------------- warm-up ------------------------------------
    with torch.no_grad():
        for _ in range(5):
            call()
    if device_obj.type == "cuda":
        torch.cuda.synchronize()

    # --------------------------- timing -------------------------------------
    start = time.perf_counter()
    with torch.no_grad():
        call()
    if device_obj.type == "cuda":
        torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000.0  # ms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="*", help=f"Models: {'|'.join(ALL_MODELS)}")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--csv", action="store_true",
                        help="Save results to results/latency_*.csv")
    args = parser.parse_args()

    targets = args.model if args.model else ALL_MODELS
    timestamp = datetime.now().isoformat(timespec="seconds")
    results: list[tuple[str, str, str, str]] = []

    for name in targets:
        try:
            mdl, dummy = load_model(name)
            ms = time_forward(mdl, dummy, args.device)
            print(f"{name:10s} | {args.device} | {ms:10.6f} ms")
            results.append((timestamp, name, args.device, f"{ms:.6f}"))
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}")
            results.append((timestamp, name, args.device, str(exc)))

    # --------------------------- CSV dump -----------------------------------
    if args.csv:
        tag = "all" if not args.model else "_".join(args.model)
        path = Path(f"results/latency_{tag}_{args.device}.csv")
        path.parent.mkdir(exist_ok=True)
        # avoid overwrite
        idx, cand = 1, path
        while cand.exists():
            cand = path.with_stem(f"{path.stem}_{idx}")
            idx += 1
        with cand.open("a", newline="") as f:
            csv.writer(f).writerows(results)
        print(f"[✓] Results saved to {cand}")


if __name__ == "__main__":
    main()

