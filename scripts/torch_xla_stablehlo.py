#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_torch_xla_stablehlo.py – PyTorch → StableHLO (torch‑xla 경로)

모델을 지정하지 않으면 `run_bench.ALL_MODELS` 전체를 시도합니다.
* Vision/CNN, LLM, GNN을 **사전에 구분**하여 example inputs를 올바른 형태로 만듭니다.
* 성공 시 `results/<model>.stablehlo.mlir/` 디렉터리(MLIR + 가중치) 저장.
* 실패 시 첫 줄 오류를 STDOUT과 `results/xla_export_log.csv` 에 기록.
"""
from __future__ import annotations

import argparse, csv, os
from pathlib import Path
from typing import Any

import torch
from torch_xla.stablehlo import (
    exported_program_to_stablehlo,
    StableHLOGraphModule,
)

from scripts.run_bench import load_model, ALL_MODELS, GNN_KEYS

os.environ.setdefault("PJRT_DEVICE", "")  # PJRT 워닝 억제

# ---------------------------------------------------------------------------
# Helper wrappers & input builders
# ---------------------------------------------------------------------------
class HFWrapper(torch.nn.Module):
    """Turns HF LLM forward(kwargs) → forward(ids, mask)"""

    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.m = backbone

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def make_inputs(name: str, dummy: Any):
    """Return *args tuple for torch.export depending on model type."""
    if name in GNN_KEYS:
        # dummy = (x, edge_index)
        return dummy  # two positional tensors

    if isinstance(dummy, tuple):  # Vision etc. → shape tuple
        return (torch.randn(*dummy),)

    # LLM dict → (ids, mask)
    return (dummy["input_ids"], dummy["attention_mask"])

# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="*", help="model keys; omit = all")
    ap.add_argument("--outdir", default="results/xla", help="save directory")
    ap.add_argument("--csv", action="store_true", help="append csv log")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)
    models = args.model if args.model else ALL_MODELS

    rows = []

    for name in models:
        try:
            print(f"[{name}] torch_xla → StableHLO export …")
            model, dummy = load_model(name)
            if isinstance(dummy, dict):
                model = HFWrapper(model)  # LLM kwargs → positional
            model.eval()

            ep = torch.export.export(model, make_inputs(name, dummy))
            shlo = exported_program_to_stablehlo(ep)

            dest = outdir / f"{name}_stablehlo"
            shlo.save(dest)
            print(f"  ✓ saved → {dest}")
            rows.append([name, "ok", ""])
        except Exception as e:
            reason = str(e).splitlines()[0]
            print(f"  [ERROR] {name}: {reason}")
            rows.append([name, "error", reason])

    if args.csv:
        log = outdir / "xla_export_log.csv"
        with log.open("a", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"[✓] log appended → {log}")


if __name__ == "__main__":
    main()

