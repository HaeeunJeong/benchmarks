#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_torch_mlir_stablehlo.py
──────────────────────────────────────────────────────────────────────────────
PyTorch  →  StableHLO  via torch-mlir.fx.export_and_import (OutputType.STABLEHLO)

* 모델은 모두 models/<name>_block.py 의
    get_model(), get_dummy_input()  래퍼로 로드한다.
* 이름을 주지 않으면 models/ 하위 모든 *_block.py 를 자동 탐색한다.
* 성공 시  results/<name>_stablehlo.torch_mlir.mlir  저장
  (텍스트 MLIR, weights 내장 아님).
* 실패 시 첫 줄 오류를 STDOUT  +  CSV(results/torch_mlir_log.csv) 기록
"""
from __future__ import annotations

import argparse, csv, importlib, os, sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch

# ─────────────────────────────────────────────────────────────────────────────
# torch-mlir  체크
# ─────────────────────────────────────────────────────────────────────────────
try:
    from torch_mlir.fx import export_and_import
    from torch_mlir.compiler_utils import OutputType
except ImportError as e:  # pragma: no cover
    sys.exit(
        "[ERROR] torch-mlir 2024+ 가 필요합니다. "
        "`pip install torch-mlir-nightly` 로 설치 후 다시 실행하세요."
    )

# ─────────────────────────────────────────────────────────────────────────────
# 모델 블록 로더
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"

def discover_model_keys() -> list[str]:
    """models/ 폴더의 *_block.py → key 추출"""
    return sorted(
        f.stem[:-6]  # strip "_block"
        for f in MODELS_DIR.glob("*_block.py")
        if f.stem.endswith("_block")
    )

def load_model_block(name: str, device: str = "cpu") -> tuple[torch.nn.Module, Any]:
    """
    models/<name>_block.py  import  →  get_model(), get_dummy_input()
    """
    mod = importlib.import_module(f"models.{name}_block")
    model = mod.get_model().to(device).eval()
    dummy = mod.get_dummy_input()
    return model, dummy

# ─────────────────────────────────────────────────────────────────────────────
# 입력 변환 (tuple / dict → positional)
# ─────────────────────────────────────────────────────────────────────────────
class HFPosWrapper(torch.nn.Module):
    """kwargs(HF) → forward(ids, mask)"""
    def __init__(self, m: torch.nn.Module):
        super().__init__(); self.m = m
    def forward(self, ids, mask):  # type: ignore
        return self.m(input_ids=ids, attention_mask=mask).last_hidden_state

def make_inputs(dummy: Any) -> tuple:
    """전달받은 dummy 를 torch-mlir export 에 넣을 *args 형태로 변환"""
    if isinstance(dummy, tuple):
        # (Tensor, Tensor) = GNN  vs  (int,int,…) shape = Vision
        if len(dummy) == 2 and isinstance(dummy[0], torch.Tensor):
            return dummy
        return (torch.randn(*dummy),)
    if isinstance(dummy, dict):  # LLM dict
        return (dummy["input_ids"], dummy["attention_mask"])
    raise RuntimeError(f"Unsupported dummy spec: {type(dummy)}")

# ─────────────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="*", help="model keys; omit = all discovered")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--outdir", default="results/torch-mlir", help="save dir")
    ap.add_argument("--csv", action="store_true", help="append csv log")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)
    keys: Iterable[str] = args.model or discover_model_keys()
    timestamp = datetime.now().isoformat(timespec="seconds")
    rows: list[list[str]] = []

    for name in keys:
        try:
            print(f"[{name}] torch-mlir export …")
            model, dummy = load_model_block(name, args.device)

            # HF LLM 은 kwargs → positional 래핑
            if isinstance(dummy, dict):
                model = HFPosWrapper(model)

            shlo_module = export_and_import(
                model, *make_inputs(dummy), output_type=OutputType.STABLEHLO
            )

            out_path = outdir / f"{name}_stablehlo.torch_mlir.mlir"
            out_path.write_text(str(shlo_module))
            print(f"   ✓ saved → {out_path}")
            rows.append([timestamp, name, "ok", ""])
        except Exception as e:
            reason = str(e).splitlines()[0]
            print(f"   [ERROR] {name}: {reason}")
            rows.append([timestamp, name, "error", reason])

    # CSV 로그
    if args.csv:
        csv_path = outdir / "torch_mlir_log.csv"
        with csv_path.open("a", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"[✓] log appended → {csv_path}")

if __name__ == "__main__":
    main()

