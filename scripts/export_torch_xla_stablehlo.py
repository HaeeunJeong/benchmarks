#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_torch_xla_stablehlo.py
──────────────────────────────────────────────────────────────────────────────
PyTorch  →  StableHLO  via torch-xla (exported_program_to_stablehlo).

* 모델을 주지 않으면 models/ 디렉터리의 모든 *_block.py 를 자동 탐색합니다.
* 성공 :  results/xla/<name>_stablehlo/   디렉터리(MLIR+weights) 저장
* 실패 :  STDOUT + CSV(results/xla_export_log.csv) 기록
"""
from __future__ import annotations

import argparse, csv, importlib, os, sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
from torch_xla.stablehlo import exported_program_to_stablehlo

# PJRT 디바이스 워닝 억제 (빈 문자열 = 런타임 default)
os.environ.setdefault("PJRT_DEVICE", "")

# ─────────────────────────────────────────────────────────────────────────────
# 모델 블록 로딩
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
    import models/<name>_block.py  →
        get_model(), get_dummy_input()
    """
    mod = importlib.import_module(f"models.{name}_block")
    model = mod.get_model().to(device).eval()
    dummy = mod.get_dummy_input()
    return model, dummy

# ─────────────────────────────────────────────────────────────────────────────
# 입력 구성 & HF 래퍼
# ─────────────────────────────────────────────────────────────────────────────
class HFWrapper(torch.nn.Module):
    """kwargs(HF) → forward(ids, mask)"""
    def __init__(self, m: torch.nn.Module):
        super().__init__(); self.m = m
    def forward(self, ids, mask):  # type: ignore
        return self.m(input_ids=ids, attention_mask=mask).last_hidden_state

def make_inputs(dummy: Any) -> tuple:
    """dummy 사양을 torch.export 가 받을 *args 로 변환"""
    if isinstance(dummy, tuple):
        # (Tensor, Tensor) = GNN  vs  shape-tuple = Vision
        if len(dummy) == 2 and all(isinstance(t, torch.Tensor) for t in dummy):
            return dummy
        return (torch.randn(*dummy),)
    if isinstance(dummy, dict):          # LLM dict
        return (dummy["input_ids"], dummy["attention_mask"])
    raise RuntimeError(f"Unsupported dummy type: {type(dummy)}")

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="*", help="model keys; omit = all discovered")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--outdir", default="results/xla", help="output directory")
    ap.add_argument("--csv", action="store_true", help="append csv log")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    keys: Iterable[str] = args.model or discover_model_keys()
    timestamp = datetime.now().isoformat(timespec="seconds")
    rows: list[list[str]] = []

    for name in keys:
        try:
            print(f"[{name}] torch_xla → StableHLO export …")
            model, dummy = load_model_block(name, args.device)

            # HF 모델이면 kwargs → positional 래퍼
            if isinstance(dummy, dict):
                model = HFWrapper(model)

            ep = torch.export.export(model, make_inputs(dummy))
            shlo = exported_program_to_stablehlo(ep)

            dest = outdir / f"{name}_stablehlo"
            shlo.save(dest)  # 디렉터리 생성 & MLIR/weights 저장
            print(f"   ✓ saved → {dest}")
            rows.append([timestamp, name, "ok", ""])
        except Exception as e:
            reason = str(e).splitlines()[0]
            print(f"   [ERROR] {name}: {reason}")
            rows.append([timestamp, name, "error", reason])

    if args.csv:
        log = outdir / "xla_export_log.csv"
        with log.open("a", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"[✓] log appended → {log}")

if __name__ == "__main__":
    main()

