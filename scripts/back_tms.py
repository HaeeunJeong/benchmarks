#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_torch_mlir_stablehlo.py – PyTorch → StableHLO via **torch-mlir**.

If no model keys are given, every model in run_bench.ALL_MODELS is attempted.
Successful exports are saved to:
    results/<model>.stablehlo.torch_mlir.mlir
Errors are logged to STDOUT and results/mlir_export_log.csv.

Requirements
------------
* torch         ≥ 2.2 (for torch.export)
* torch-mlir    ≥ 2024.0
* Set env var `TORCH_MLIR_ENABLE_STABLEHLO_LOWERING=1` if your build requires it.
"""
from __future__ import annotations

import argparse, csv
from pathlib import Path
from datetime import datetime
from typing import Any

import torch
try:
    from torch_mlir.fx import export_and_import
    from torch_mlir.compiler_utils import OutputType
except ImportError:
    raise ImportError("torch-mlir is not installed. Please install it via pip: `pip install torch-mlir`.")

from scripts.run_bench import load_model, ALL_MODELS

# ---------------------------------------------------------------------------
# prepare positional example inputs (same logic as torch_xla script)
# ---------------------------------------------------------------------------

def make_inputs(dummy: Any):
    if isinstance(dummy, tuple):
        if len(dummy) == 2 and isinstance(dummy[0], torch.Tensor):
            # GNN (x, edge_index)
            return dummy
        # Vision: shape tuple
        return (torch.randn(*dummy),)
    if isinstance(dummy, dict):
        return (dummy["input_ids"], dummy["attention_mask"])
    raise RuntimeError("Unknown dummy spec")

def export_to_stablehlo(model, *inputs):
    mlir_module = export_and_import(model, *inputs, output_type=OutputType.STABLEHLO)
    return mlir_module

# HF wrapper for positional ids/mask
class HFPosWrapper(torch.nn.Module):
    def __init__(self, mdl: torch.nn.Module):
        super().__init__()
        self.m = mdl
    def forward(self, ids, mask):  # noqa: D401
        return self.m(input_ids=ids, attention_mask=mask).last_hidden_state

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="*", help="model keys; empty = ALL")
    ap.add_argument("--outdir", default="results/torch-mlir", help="dir to save .mlir files")
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)
    models = args.model if args.model else ALL_MODELS

    rows = []
    stamp = datetime.now().isoformat(timespec="seconds")

    for name in models:
        try:
            print(f"[{name}] torch-mlir compile → stablehlo …")
            mdl, dummy = load_model(name)
            mdl.eval()

            # Wrap HF models so they accept positional inputs
            if isinstance(dummy, dict):
                mdl = HFPosWrapper(mdl)

            module = export_to_stablehlo(mdl, make_inputs(dummy))

            path = outdir / f"{name}_stablehlo"
            path.write_text(str(module))
            print(f"  ✓ saved → {path}")
            rows.append([stamp, name, "ok"])
        except Exception as e:
            reason = str(e).splitlines()[0]
            print(f"  [ERROR] {name}: {reason}")
            rows.append([stamp, name, f"error: {reason}"])

    if args.csv:
        log = outdir / "mlir_export_log.csv"
        with log.open("a", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"[✓] CSV appended → {log}")

if __name__ == "__main__":
    main()

