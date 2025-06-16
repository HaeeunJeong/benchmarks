#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""iree_run.py – Compile models/* blocks to IREE vmfb (weights externalized).

* 파라미터는 `.safetensors` 로 저장하고, IR에는 `dense_resource` 참조만 남김.
* vmfb 실행 시 `iree.runtime.ParameterIndex` 로 메모리 상에 가중치를 주입.
* 결과 artefacts (예: conv):
    results/iree/conv/
      ├─ module.vmfb            (compiled binary)
      ├─ params.safetensors     (weights)
      ├─ input.npy              (샘플 입력, 재현용)
      └─ latency.txt            (1‑shot ms)
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

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_inputs(dummy: Any):
    if isinstance(dummy, tuple):
        if len(dummy) == 2 and isinstance(dummy[0], torch.Tensor):
            return dummy  # (x, edge_index)
        return (torch.randn(*dummy),)
    if isinstance(dummy, dict):
        return (dummy["input_ids"], dummy["attention_mask"])
    return (dummy,)


# def export_and_externalize(model: torch.nn.Module, inputs):
#     weights: Dict[str, torch.Tensor] = {n: p.detach().contiguous() for n, p in model.named_parameters()}
#     aot.externalize_module_parameters(model)
#     exported = aot.export(model, *inputs)
#     return exported, weights

def export_and_externalize(model: torch.nn.Module, inputs):
    # 1) 모든 파라미터 + 버퍼를 dict 에 복사
    # weights = {n: p.detach().contiguous()
    #            for n, p in model.named_parameters()}
    # buffers = {n: b.detach().contiguous()
    #            for n, b in model.named_buffers()}
    # weights.update(buffers)                # <-- BN running_mean/var 포함

    weights = {n: p.detach().contiguous() for n, p in model.named_parameters()}
    for n, b in model.named_buffers():
        if not b.requires_grad:        # 모든 buffer
            weights[n] = b.detach().contiguous()

    # 2) 외부화
    aot.externalize_module_parameters(model)
    exported = aot.export(model, *inputs)
    return exported, weights


def save_weights_safetensor(weights: Dict[str, torch.Tensor], path: Path):
    save_file(weights, str(path))


# ---------------------------------------------------------------------------
# IREE run helpers
# ---------------------------------------------------------------------------

def run_vmfb(vmfb_mm, backend: str, inputs, weights: Dict[str, torch.Tensor]):
    """vmfb_mm: MappedMemory from exported.compile()"""
    drv_map = rt.TARGET_BACKEND_TO_DRIVER
    driver = backend if backend in rt.query_available_drivers() else drv_map.get(backend, "llvm")

    # Build ParameterIndex (in‑memory buffers)
    idx = ireert.ParameterIndex()
    for k, t in weights.items():
        idx.add_buffer(k, t.numpy().tobytes())

    cfg = ireert.Config(driver_name=driver)
    inst = cfg.vm_instance

    param_module = ireert.create_io_parameters_module(inst, idx.create_provider(scope="model"))
    hal_module = ireert.create_hal_module(inst, cfg.device)
    main_module = ireert.VmModule.copy_buffer(inst, vmfb_mm.map_memory())

    modules = ireert.load_vm_modules(param_module, hal_module, main_module, config=cfg)
    vm = modules[-1]

    np_inputs = [t.detach().cpu().numpy() for t in inputs]
    res = vm.main(*np_inputs)
    if hasattr(res, "to_host"):
        return res.to_host()
    return res


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="*", help="block keys; omit=all")
    ap.add_argument("--backend", default="llvm-cpu", choices=["llvm-cpu", "cuda", "vulkan", "rocm"])
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--mode", default="aot", choices=["aot", "jit"],
               help="aot: externalize weights (default) | jit: embed weights")
    args = ap.parse_args()

    models = args.model if args.model else ALL_MODELS
    base_dir = Path("results/iree"); base_dir.mkdir(exist_ok=True)

    rows = []; ts = datetime.now().isoformat(timespec="seconds")

    for name in models:
        try:
            print(f"[{name}] export → compile → run …")
            mdl, dummy = load_model(name); mdl.eval()
            inp = make_inputs(dummy)

            if args.mode == "aot":
                exported, weight_dict = export_and_externalize(mdl, inp)
            else: # args.mode == "jit"
                exported = aot.export(mdl, *inp)
                weight_dict = {}

            mdl_dir = base_dir / name; mdl_dir.mkdir(exist_ok=True)
            # safepath = mdl_dir / "params.safetensors"; save_weights_safetensor(weight_dict, safepath)
            np.save(mdl_dir / "input.npy", inp[0].detach().cpu().numpy())
            print(f"  ✓ exported {name} with {len(weight_dict)} weights")

            # vmfb_mm_for_save = exported.compile(save_to=f"{mdl_dir}/{name}-{args.backend}.vmfb", target_backends=[args.backend])
            vmfb_mm = exported.compile(save_to=None, target_backends=[args.backend])
            # print(f"  ✓ saved weights to {safepath}")

            st = time.perf_counter(); run_vmfb(vmfb_mm, args.backend, inp, weight_dict); lat = (time.perf_counter()-st)*1000
            (mdl_dir / "latency.txt").write_text(f"{lat:.3f} ms\n")
            print(f"  ✓ {lat:.3f} ms  → {mdl_dir}")
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

