#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compile_iree.py – Compile supported PyTorch models to IREE vmfb & run.

*   Exports model → StableHLO via `torch.export` + torch‑xla helper
*   Compiles StableHLO → vmfb using the embedded `iree-compile` Python API
*   Runs vmfb once on the requested HAL backend (llvm-cpu, cuda, …)
*   Appends results to `results/iree_latency*.csv` (time | error | unsupported)

Requires the `stablehlo-iree.yaml` (or *_gpu) env from /env to be activated.
Python 3.11.  
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import torch
from iree import runtime as ireert
from iree.compiler import compile_str
from torch_xla.stablehlo import exported_program_to_stablehlo

# Re‑use model list / loader from run_bench
from scripts.run_bench import load_model, ALL_MODELS, GNN_KEYS

# ---------------------------------------------------------------------------
# Configurable “unsupported” set – ops not yet captured by PyTorch → StableHLO
# ---------------------------------------------------------------------------
UNSUPPORTED: set[str] = {
    # PyG scatters/nonzero currently not legalised for StableHLO
    *GNN_KEYS,
    # Large HF transformers need custom kwargs; skip by default
    "bert", "gpt2", "llama", "deepseek",
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _to_stablehlo(model: torch.nn.Module, example) -> str:
    """Return StableHLO MLIR string."""
    model.eval()
    if isinstance(example, tuple):
        exported = torch.export.export(model, example)
    else:
        exported = torch.export.export(model, (), kwargs=example)
    shlo = exported_program_to_stablehlo(exported)
    # `shlo.mlir_module` is an MLIR object; str() gives textual format
    return str(shlo.mlir_module)


def _compile_vmfb(hlo_text: str, target: str) -> bytes:
    flags = [f"-iree-target-backends={target}", "--iree-stream-resource-index-bits=64"]
    return compile_str(hlo_text, target_backends=target, extra_args=flags)


def _run_vmfb(blob: bytes) -> float:
    cfg = ireert.Config("local-task")
    ctx = ireert.RuntimeContext(cfg)
    mod = ireert.VmModule.from_flatbuffer(ctx.instance, blob)
    ctx.register_vm_module(mod)
    fn = ctx.modules.module["main"]
    start = time.perf_counter(); fn(); return (time.perf_counter() - start) * 1000

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="*", help="model keys (omit for ALL)")
    ap.add_argument("--target", default="llvm-cpu", choices=["llvm-cpu", "cuda", "vulkan", "rocm"], help="IREE HAL back‑end")
    ap.add_argument("--csv", action="store_true", help="save csv to results/")
    args = ap.parse_args()

    models: list[str] = args.model if args.model else ALL_MODELS
    stamp = datetime.now().isoformat(timespec="seconds")
    rows: list[list[str]] = []

    for name in models:
        if name in UNSUPPORTED:
            print(f"[skip] {name}: unsupported")
            rows.append([stamp, name, args.target, "unsupported"])
            continue
        try:
            print(f"[{name}] export → compile → run …")
            mdl, dummy = load_model(name)
            hlo = _to_stablehlo(mdl, dummy)
            vmfb = _compile_vmfb(hlo, args.target)
            ms = _run_vmfb(vmfb)
            print(f"✓ {name} : {ms:.2f} ms")
            # save vmfb
            Path("results").mkdir(exist_ok=True)
            (Path("results") / f"{name}-{args.target}.vmfb").write_bytes(vmfb)
            rows.append([stamp, name, args.target, f"{ms:.2f}"])
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            rows.append([stamp, name, args.target, f"error: {e}"])

    if args.csv:
        tag = "all" if not args.model else "_".join(args.model)
        out = Path(f"results/iree_latency_{tag}.csv")
        out.parent.mkdir(exist_ok=True)
        idx = 1
        cand = out
        while cand.exists():
            cand = out.with_stem(f"{out.stem}_{idx}")
            idx += 1
        with cand.open("a", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"[✓] CSV appended → {cand}")


if __name__ == "__main__":
    main()




