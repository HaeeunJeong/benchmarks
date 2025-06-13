#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_torch_mlir_stablehlo.py – PyTorch → StableHLO via **torch‑mlir.fx.export_and_import**

*   torch‑mlir 2024+ 는 고수준 `torch_mlir.compile()` 대신
    `torch_mlir.fx.export_and_import(..., output_type=OutputType.STABLEHLO)` 사용.
*   모델 이름을 주지 않으면 `run_bench.ALL_MODELS` 전체를 시도.
*   성공 시  `results/<model>.stablehlo.torch_mlir.mlir`  텍스트 저장.
*   실패 시 첫 줄 오류를 STDOUT + CSV(`results/mlir_export_log.csv`) 기록.
"""
from __future__ import annotations

import argparse, csv, os
from pathlib import Path
from datetime import datetime
from typing import Any

import torch

try:
    from torch_mlir.fx import export_and_import
    from torch_mlir.compiler_utils import OutputType
except ImportError as e:
    raise RuntimeError("torch-mlir 2024+ 설치 필요 (`pip install torch-mlir-nightly`)" ) from e

from scripts.run_bench import load_model, ALL_MODELS

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class HFPosWrapper(torch.nn.Module):
    """Wrap HF model so it takes (ids, mask) positional args."""
    def __init__(self, m: torch.nn.Module):
        super().__init__(); self.m = m
    def forward(self, ids, mask):  # type: ignore
        return self.m(input_ids=ids, attention_mask=mask).last_hidden_state

def make_inputs(dummy: Any):
    """Return positional inputs matching each sub‑class."""
    if isinstance(dummy, tuple):
        # GNN (x, edge_index)  vs  Vision(shape)
        if len(dummy) == 2 and isinstance(dummy[0], torch.Tensor):
            return dummy
        return (torch.randn(*dummy),)
    if isinstance(dummy, dict):
        return (dummy["input_ids"], dummy["attention_mask"])
    raise RuntimeError("Unsupported dummy spec")


# ---------------------------------------------------------------------------
# post‑export clean‑ups (optional)
# ---------------------------------------------------------------------------
from torch_mlir.compiler_utils import run_pipeline_with_repro_report

def postprocess(mod):
        # ① assert·shape 계산 제거
    run_pipeline_with_repro_report(
        mod,
        "torch-drop-abstract-interp-calculations",
    )
    # ② canonicalize + stablehlo 정리 (상수 folding·중복 제거)
    run_pipeline_with_repro_report(mod, "canonicalize", "canon")
    run_pipeline_with_repro_report(mod, "stablehlo-aggressive-simplification", "shlo simp")
    return mod

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="*", help="model keys; empty = ALL")
    ap.add_argument("--outdir", default="results/torch-mlir", help="output directory")
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)
    models = args.model if args.model else ALL_MODELS
    rows:list[list[str]] = []
    stamp = datetime.now().isoformat(timespec="seconds")

    for name in models:
        try:
            print(f"[{name}] torch-mlir.fx export → stablehlo …")
            mdl, dummy = load_model(name); mdl.eval()
            if isinstance(dummy, dict):
                mdl = HFPosWrapper(mdl)

            shlo_mod = export_and_import(
                mdl, *make_inputs(dummy), output_type=OutputType.STABLEHLO
            )
            postprocess(shlo_mod)

            # print(shlo_mod)

            out = outdir / f"{name}_stablehlo.mlir"
            out.write_text(str(shlo_mod))
            print(f"  ✓ saved → {out}")
            rows.append([stamp, name, "ok"])
        except Exception as e:
            reason = str(e).splitlines()[0]
            print(f"  [ERROR] {name}: {reason}")
            rows.append([stamp, name, f"error: {reason}"])

    if args.csv:
        csv_path = outdir / "torch-mlir_log.csv"
        with csv_path.open("a", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"[✓] CSV appended → {csv_path}")

if __name__ == "__main__":
    main()

