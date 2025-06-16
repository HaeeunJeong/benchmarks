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
from torch._subclasses.fake_tensor import FakeTensorMode

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
from torch_mlir.compiler_utils import PassManager as pm
from torch_mlir.compiler_utils import run_pipeline_with_repro_report

def postprocess(mod):
    """Try to remove asserts / split big constants so lowering succeeds."""
    try:
        run_pipeline_with_repro_report(
            mod, "torch-suppress-runtime-asserts", "suppress runtime assert")
        run_pipeline_with_repro_report(
            mod, "torch-split-large-constants", "split constants")
    except Exception as e:
        # Non‑fatal: just skip if pass unavailable
        print(f"    (postprocess skipped: {e.splitlines()[0]})")
    return mod

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _load_empty_hf(model_name: str):
    """Load HF model **structure only**: no weights (uses meta device)."""
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(model_name)
    mdl = AutoModel.from_config(cfg)
    return mdl.to(torch.device("meta"))

def _replace_leaf(mod, qname, obj, *, is_param):
    parts, leaf = qname.split("."), qname.split(".")[-1]
    tgt = mod
    for p in parts[:-1]:
        tgt = getattr(tgt, p)
    if is_param:
        tgt.register_parameter(leaf, torch.nn.Parameter(obj, requires_grad=False))
    else:
        tgt.register_buffer(leaf, obj)


# def _replace_leaf(mod: torch.nn.Module, qual_name: str, obj, *, is_param: bool):
#     """qual_name = 'layer1.0.weight' → 찾아가서 교체"""
#     parts = qual_name.split(".")
#     tgt_mod = mod
#     for p in parts[:-1]:
#         tgt_mod = getattr(tgt_mod, p)
#     leaf = parts[-1]
#     if is_param:
#         tgt_mod.register_parameter(leaf, obj)
#     else:
#         tgt_mod.register_buffer(leaf, obj)

def empty_parameters(m: torch.nn.Module):
    """move every param/buffer to meta device (structure-only)."""
    for n, p in list(m.named_parameters(recurse=True)):
        meta_p = torch.empty_like(p, device="meta", requires_grad=p.requires_grad)
        _replace_leaf(m, n, meta_p, is_param=True)
    for n, b in list(m.named_buffers(recurse=True)):
        meta_b = torch.empty_like(b, device="meta")
        _replace_leaf(m, n, meta_b, is_param=False)
    return m



# def empty_parameters(m: torch.nn.Module):
#     """Move all parameters/buffers to meta device to free RAM."""
#     for n, p in list(m.named_parameters(recurse=True)):
#         with torch.no_grad():
#             new_p = torch.empty_like(p, device="meta")
#             setattr(m, n, torch.nn.Parameter(new_p, requires_grad=p.requires_grad))
#     for n, b in list(m.named_buffers(recurse=True)):
#         new_b = torch.empty_like(b, device="meta")
#         setattr(m, n, new_b)
#     return m


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

            # ---- Remove parameter tensors to save memory ----
            if isinstance(dummy, dict):  # HF LLM: load empty structure instead
                mdl = _load_empty_hf(HF_MODELS[name]) if name in HF_MODELS else mdl
            mdl = empty_parameters(mdl)

            if isinstance(dummy, dict):
                mdl = HFPosWrapper(mdl)

            with FakeTensorMode():
                shlo_mod = export_and_import(
                    mdl, *make_inputs(dummy), output_type=OutputType.STABLEHLO
                )

            postprocess(shlo_mod)

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

