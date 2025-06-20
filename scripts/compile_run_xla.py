#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compile_xla.py – Run benchmarks on **PyTorch XLA back‑end**.

* Mirrors the CLI/output format of compile_iree.py but executes the model on an
  XLA device (TPU, XLA‑CPU, or GPU‑via‑PJRT) instead of compiling with IREE.
* Measures **first‑step latency** after graph compilation (warm‑up + mark_step).
* Saves per‑run CSV to results/xla_latency_<tag>.csv and prints stdout summary.

Example
-------
$ python -m scripts.compile_xla conv resnet --device xla
$ python -m scripts.compile_xla             # all models, default xla:cpu
"""
from __future__ import annotations

import argparse, csv, os, time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch_xla
import torch_xla.core.xla_model as xm

from scripts.run_bench import load_model, ALL_MODELS

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_inputs(dummy: Any, device):
    if isinstance(dummy, tuple):
        if len(dummy) == 2 and isinstance(dummy[0], torch.Tensor):
            return tuple(t.to(device) for t in dummy)
        return (torch.randn(*dummy, device=device),)
    if isinstance(dummy, dict):
        return {k: v.to(device) for k, v in dummy.items()}
    return (dummy.to(device),)


def run_one(model: torch.nn.Module, dummy: Any, *, device: str) -> float:
    model.to(device).train(False)
    inputs = make_inputs(dummy, device)

    def _call():
        with torch.no_grad():
            if isinstance(inputs, dict):
                _ = model(**inputs)
            elif isinstance(inputs, tuple):
                _ = model(*inputs)
            else:
                _ = model(inputs)

    # compile graph – first step
    _call(); xm.mark_step()
    # warm‑up 4 more
    for _ in range(4):
        _call(); xm.mark_step()

    # timed step
    t0 = time.perf_counter()
    _call(); xm.mark_step()
    xm.wait_device_ops()  # sync
    return (time.perf_counter() - t0) * 1000.0  # ms

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="*", help="model keys; omit for ALL")
    ap.add_argument("--device", default="xla:1", help="xm.xla_device() spec, e.g. 'xla', 'xla:1'")
    ap.add_argument("--csv", action="store_true", help="write results/xla_latency_*.csv")
    args = ap.parse_args()

    models = args.model if args.model else ALL_MODELS
    device = xm.xla_device() if args.device == "xla:1" else torch.device(args.device)
    out_rows: list[list[str]] = []
    ts = datetime.now().isoformat(timespec="seconds")

    for name in models:
        try:
            mdl, dummy = load_model(name)
            ms = run_one(mdl, dummy, device=device)
            print(f"{name:10s} | {device} | {ms:9.3f} ms")
            out_rows.append([ts, name, str(device), f"{ms:.3f}"])
        except Exception as e:
            reason = str(e).splitlines()[0]
            print(f"[ERROR] {name}: {reason}")
            out_rows.append([ts, name, str(device), f"error: {reason}"])

    if args.csv:
        tag = "all" if not args.model else "_".join(args.model)
        csv_path = Path("results/xla_latency_" + tag + ".csv")
        csv_path.parent.mkdir(exist_ok=True)
        idx, cand = 1, csv_path
        while cand.exists():
            cand = csv_path.with_stem(csv_path.stem + f"_{idx}"); idx += 1
        with cand.open("a", newline="") as f:
            csv.writer(f).writerows(out_rows)
        print(f"[✓] CSV saved → {cand}")

if __name__ == "__main__":
    os.environ.setdefault("PJRT_DEVICE", "")  # silence PJRT warning noise
    main()

