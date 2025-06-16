#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""iree_run.py – Compile models/* blocks to IREE vmfb **with external params**.

*   Uses **iree‑turbine aot** (`aot.export`) so that weight tensors are saved
    separately as `.safetensors` (or `.npy`) files, keeping the StableHLO IR
    tiny and avoiding OOM during compile.
*   Saves artefacts to `results/iree/<model>/`:
      ├─ module.vmfb           (compiled binary)
      └─ params.safetensors    (weights)
*   Logs latency to CSV (single inference on the chosen backend).
"""
from __future__ import annotations

import argparse, csv, os, time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import numpy as np
from safetensors.torch import save_file

import iree.turbine.aot as aot
import iree.runtime as ireert
import iree.runtime as rt

from scripts.run_bench import load_model, ALL_MODELS

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def make_inputs(dummy: Any):
    """Convert run_bench dummy → positional tuple of tensors (np arrays later)."""
    if isinstance(dummy, tuple):
        if len(dummy) == 2 and isinstance(dummy[0], torch.Tensor):
            return dummy  # (x, edge_index)
        return (torch.randn(*dummy),)  # Vision: shape tuple
    if isinstance(dummy, dict):
        return (dummy["input_ids"], dummy["attention_mask"])
    return (dummy,)  # single tensor


def export_with_external_params(model: torch.nn.Module, inputs):
    """Externalize params → safetensors, export IR."""
    # 1. save params as safetensors dict
    weights: Dict[str, torch.Tensor] = {
        n: p.detach().clone().contiguous() for n, p in model.named_parameters()
    }
    # 2. externalize such that IR has symbolic resource refs
    aot.externalize_module_parameters(model)
    exported = aot.export(model, *inputs)
    return exported, weights


# -----------------------------------------------------------------------------
# IREE helpers
# -----------------------------------------------------------------------------

# def compile_vmfb(exported, backend: str):
#     print(exported)  # debug: print the exported StableHLO IR
#     return exported.compile(save_to="iree.vmfb", target_backends=[backend])


def run_vmfb(vmfb: bytes, backend: str, inputs):
    print("1")
    drv_map = rt.TARGET_BACKEND_TO_DRIVER
    print("2")
    driver = backend if backend in rt.query_available_drivers() else drv_map.get(backend, "llvm")
    print(f"Using driver: {driver}")
    mod = ireert.load_vm_flatbuffer(vmfb, driver=driver)
    print("3")
    np_inputs = [t.detach().cpu().numpy() for t in inputs]
    print("4")
    out = mod.main(*np_inputs)
    print("5")
    return out.to_host() if hasattr(out, "to_host") else out


# -----------------------------------------------------------------------------
# main CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="*", help="block keys; omit=all")
    ap.add_argument("--backend", default="llvm-cpu", choices=["llvm-cpu", "cuda", "vulkan", "rocm"])
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    models = args.model if args.model else ALL_MODELS
    base_dir = Path("results/iree"); base_dir.mkdir(parents=True, exist_ok=True)

    rows = []; ts = datetime.now().isoformat(timespec="seconds")

    for name in models:
        try:
            print(f"[{name}] export → compile → run …")
            mdl, dummy = load_model(name); mdl.eval()
            inp = make_inputs(dummy)
            print("Exporting with external params …")
            exported, weight_dict = export_with_external_params(mdl, inp)

            model_dir = base_dir / name; model_dir.mkdir(exist_ok=True)
            # save weights
            safepath = model_dir / "params.safetensors"; save_file(weight_dict, str(safepath))

            # compile vmfb
            # print(f"Compiling {args.backend} vmfb …")
            # vmfb = compile_vmfb(exported, args.backend)
            #
            # print(f"Saving vmfb to {model_dir} …")
            # (model_dir / f"{name}-{args.backend}.vmfb").write_bytes(vmfb)

            print(f"Compile and save {args.backend} vmfb …")
            vmfb = exported.compile(save_to=f"{name}-{args.backend}.vmfb", target_backends=[args.backend])

            # latency run
            print(f"Running {args.backend} vmfb …")
            st = time.perf_counter(); run_vmfb(vmfb, args.backend, inp); lat = (time.perf_counter()-st)*1000
            print(f"  ✓ {lat:.3f} ms | {model_dir}")
            rows.append([ts, name, args.backend, f"{lat:.3f}"])
        except Exception as e:
            reason = str(e).splitlines()[0]
            print(f"  [ERROR] {name}: {reason}")
            rows.append([ts, name, args.backend, f"error: {reason}"])

    if args.csv:
        tag = "all" if not args.model else "_".join(args.model)
        csv_p = base_dir / f"iree_latency_{tag}_{args.backend}.csv"
        with csv_p.open("a", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"[✓] CSV → {csv_p}")


if __name__ == "__main__":
    os.environ.setdefault("PJRT_DEVICE", "")  # mute torch_xla PJRT warning
    main()

