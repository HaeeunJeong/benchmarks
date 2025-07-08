#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_torch_vs_xla.py
Python 3.11

* Loads each model block defined in run_bench.py
* Benchmarks forward-pass latency on:
    1. Native PyTorch backend   (cuda / cpu)
    2. PyTorch-XLA backend      (xla:0 … TPU / PJRT GPU)
  – Warm-up 5 runs, then measure 10 runs and report the average (ms).
* Compares the final outputs from both backends (MAE).
* Writes combined results to results/latency_compare_<tag>.csv
"""

from __future__ import annotations
import argparse, csv, os, time, sys
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

import torch
import torch_xla
import torch_xla.core.xla_model as xm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  # add parent to PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))  # add scripts to PYTHONPATH

from scripts.run_bench import load_model, ALL_MODELS

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
print(f"ROOT_DIR: {ROOT_DIR}")
MODEL_DIR   = ROOT_DIR / "models"
UTIL_DIR    = ROOT_DIR / "utils"
RESULTS_DIR = ROOT_DIR / "results" / "xla"

OUT_DIR = RESULTS_DIR / "xla_out"

from scripts.run_bench import load_model, ALL_MODELS

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def prepare_inputs(dummy: Any, device: torch.device | str):
    """Move dummy-input to device (or create on device) exactly once."""
    dev = torch.device(device)
    if isinstance(dummy, torch.Tensor):
        return dummy.to(dev)
    if isinstance(dummy, tuple):
        if all(isinstance(t, torch.Tensor) for t in dummy):
            return tuple(t.to(dev) for t in dummy)
        # tuple of ints → shape
        return torch.randn(*dummy, device=dev)
    if isinstance(dummy, dict):
        return {k: v.to(dev) for k, v in dummy.items()}
    raise TypeError("Unsupported dummy_input type")

def forward_call(model: torch.nn.Module, inputs: Any):
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, tuple):
        return model(*inputs)
    return model(inputs)

def run_backend(model: torch.nn.Module,
                dummy: Any,
                backend: str = "torch",
                warmup: int = 5,
                iters: int = 10) -> Tuple[float, torch.Tensor]:
    """
    Returns (avg_ms, sample_output) for the chosen backend.
      backend = "torch" | "xla"
    """
    if backend == "torch":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Running on {device.type} backend...")
        model = model.to(device).eval()
        inputs = prepare_inputs(dummy, device)

        # warm-up
        with torch.no_grad():
            for _ in range(warmup):
                _ = forward_call(model, inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # measure
        times = []
        with torch.no_grad():
            for _ in range(iters):
                start = time.perf_counter()
                out = forward_call(model, inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1e3)
        avg_ms = sum(times) / len(times)
        return avg_ms, out.detach().cpu()

    # --------------------------- XLA backend --------------------------------
    elif backend == "xla":
        device = xm.xla_device()
        # print(f"Running on {device} backend...")
        model = model.to(device).eval()
        inputs = prepare_inputs(dummy, device)

        def _call():
            with torch.no_grad():
                return forward_call(model, inputs)

        # compile + warm-up
        for _ in range(warmup):
            out = _call(); xm.mark_step()
        xm.wait_device_ops()

        # measure
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            out = _call(); xm.mark_step()
            xm.wait_device_ops()
            times.append((time.perf_counter() - start) * 1e3)
        avg_ms = sum(times) / len(times)
        return avg_ms, out.cpu()

    else:
        raise ValueError(f"Unknown backend '{backend}'")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="*", help="model keys; omit for ALL")
    ap.add_argument("--csv", action="store_true", help="write CSV summary")
    args = ap.parse_args()

    models = args.model if args.model else ALL_MODELS
    ts = datetime.now().isoformat(timespec="seconds")
    rows = [["timestamp","model","torch_ms","xla_ms","mae"]]

    for name in models:
        try:
            mdl, dummy = load_model(name)

            torch_ms, torch_out = run_backend(mdl, dummy, "torch")
            xla_ms,   xla_out   = run_backend(mdl, dummy, "xla")

            mae = torch.nn.functional.l1_loss(
                torch_out.float(), xla_out.float()).item()

            print(f"{name:10s} | torch {torch_ms:8.3f} ms | "
                  f"xla {xla_ms:8.3f} ms | MAE {mae:.3e}")
            rows.append([ts, name, f"{torch_ms:.3f}", f"{xla_ms:.3f}", f"{mae:.3e}"])

        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            rows.append([ts, name, "error", "error", str(e)])

    # --------------------------- CSV dump -----------------------------------
    if args.csv:
        tag = "all" if not args.model else "_".join(args.model)
        p = Path(f"results/latency_compare_{tag}.csv")
        p.parent.mkdir(exist_ok=True)
        idx, cand = 1, p
        while cand.exists():
            cand = p.with_stem(f"{p.stem}_{idx}"); idx += 1
        with cand.open("w", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"[✓] CSV saved → {cand}")

if __name__ == "__main__":
    # suppress PJRT spam
    os.environ.setdefault("PJRT_DEVICE", "")
    main()

