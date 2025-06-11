#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compile_iree.py – PyTorch ➜ StableHLO ➜ IREE compile & run timer
Python 3.11
"""
import argparse
import time
import tempfile
from pathlib import Path
from datetime import datetime
import csv

from iree import runtime as ireert
from iree.compiler import compile_str
from torch_xla.stablehlo import exported_program_to_stablehlo
import torch

# run_bench.py 에서 공유한다고 가정
from scripts.run_bench import load_model, ALL_MODELS

def pytorch_to_stablehlo(model, dummy):
    model.eval()
    if isinstance(dummy, tuple):
        exported = torch.export.export(model, (dummy[0], dummy[1])) if isinstance(dummy[0], torch.Tensor) else torch.export.export(model, (torch.randn(*dummy),))
    else:
        exported = torch.export.export(model, (), kwargs=dummy)
    hlo_obj = exported_program_to_stablehlo(exported)
    # return hlo_obj.mlir_module_text
    return str(hlo_obj)

def stablehlo_to_vmfb(mlir_txt, target):
    flags = [
        f"-iree-hal-target-backends={target}",
        "--iree-stream-resource-index-bits=64",
        "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu",
    ]
    return compile_str(mlir_txt.encode("utf-8"), target_backends=target, extra_args=flags)

def run_vmfb(vmfb_blob):
    config = ireert.Config("local-task")
    ctx = ireert.RuntimeContext(config)
    mod = ireert.VmModule.from_flatbuffer(ctx.instance, vmfb_blob)
    ctx.register_vm_module(mod)
    fn = ctx.modules.module["main"]
    start = time.perf_counter()
    _ = fn()
    return (time.perf_counter() - start) * 1000  # ms

def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="*", help="Models to compile: conv | gat | resnet | mobilenet | vit | bert | gpt2 | llama | deepseek")
    p.add_argument("--target", default="llvm-cpu", choices=["llvm-cpu", "cuda", "vulkan", "rocm"])
    p.add_argument("--csv", action="store_true", help="Save results to results/iree_latency.csv")
    args = p.parse_args()

    models_to_run = args.model if args.model else ALL_MODELS
    results = []
    timestamp = datetime.now().isoformat()

    for model_name in models_to_run:
        try:
            print(f"\n[✓] Processing {model_name} → {args.target}")
            model, dummy = load_model(model_name)

            print(f"  ⤷ Exporting to StableHLO …")
            hlo_txt = pytorch_to_stablehlo(model, dummy)

            print(f"  ⤷ Compiling to IREE FlatBuffer …")
            vmfb_blob = stablehlo_to_vmfb(hlo_txt, args.target)

            print(f"  ⤷ Running on IREE runtime …")
            ms = run_vmfb(vmfb_blob)
            print(f"{model_name} on {args.target}: {ms:.2f} ms")

            # Save vmfb file
            out_path = Path("results") / f"{model_name}-{args.target}.vmfb"
            out_path.parent.mkdir(exist_ok=True)
            out_path.write_bytes(vmfb_blob)

            results.append((timestamp, model_name, args.target, ms))
        except Exception as e:
            print(f"[ERROR] {model_name}: {e}")
            results.append((timestamp, model_name, args.target, "error"))

    if args.csv:
        csv_path = Path("results/iree_latency.csv")
        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(results)
        print(f"[✓] Results saved to {csv_path}")

if __name__ == "__main__":
    main()


