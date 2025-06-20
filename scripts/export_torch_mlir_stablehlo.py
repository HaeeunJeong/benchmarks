#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""iree_run.py – Compile models/* blocks to IREE vmfb (weights optional).

* `--mode aot` (기본): 파라미터를 `.safetensors` 로 분리, IR 은 `dense_resource`
  참조만 유지.
* `--mode jit`        : 파라미터를 IR 에 인라인.
* 컴파일 전-후 모든 **중간 MLIR** (`input.mlir`, pass별 IR 등) 을
  `results/compile_iree/<model>/<backend>/` 폴더에 저장.
* 실행 한 번 후 **latency (ms)** 와 **backend** 를 CSV(`results/iree_latency...`)에
  append.
"""
from __future__ import annotations

import argparse, csv, dataclasses, os, time, traceback, subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

import torch, numpy as np
from safetensors.torch import save_file

import iree.turbine.aot as aot
import iree.runtime as ireert
import iree.runtime as rt
from iree.compiler import compile_str, CompilerOptions, InputType, OutputFormat

from scripts.run_bench import load_model, ALL_MODELS


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_inputs(dummy: Any):
    if isinstance(dummy, tuple):
        if len(dummy) == 2 and isinstance(dummy[0], torch.Tensor):
            return dummy  # (x, edge_index)  – GNN
        return (torch.randn(*dummy),)       # Vision (shape tuple)
    if isinstance(dummy, dict):             # LLM
        return (dummy["input_ids"], dummy["attention_mask"])
    return (dummy,)


def export_and_externalize(model: torch.nn.Module, inputs: Sequence[torch.Tensor]):
    """Return (ExportOutput, weight dict) after aot.externalize()."""
    weights: Dict[str, torch.Tensor] = {
        **{n: p.detach().contiguous() for n, p in model.named_parameters()},
        **{n: b.detach().contiguous()
           for n, b in model.named_buffers()}
    }
    aot.externalize_module_parameters(model)
    return aot.export(model, *inputs), weights


def save_weights(weights: Dict[str, torch.Tensor], path: Path):
    if weights:
        save_file(weights, str(path))


# ─────────────────────────────────────────────────────────────────────────────
# IREE runtime helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_param_index(weights: Dict[str, torch.Tensor]):
    idx = ireert.ParameterIndex()
    for k, t in weights.items():
        idx.add_buffer(k, t.detach().numpy().tobytes())
    return idx


def run_vmfb(vmfb_mm, backend: str, inputs, weights: Dict[str, torch.Tensor]):
    drv_map = rt.TARGET_BACKEND_TO_DRIVER
    driver = backend if backend in rt.query_available_drivers() else drv_map.get(backend, "llvm")

    cfg = ireert.Config(driver_name=driver)
    inst = cfg.vm_instance

    modules = []
    if weights:
        param_mod = ireert.create_io_parameters_module(
            inst, build_param_index(weights).create_provider(scope="model"))
        modules.append(param_mod)

    modules.append(ireert.create_hal_module(inst, cfg.device))
    modules.append(ireert.VmModule.copy_buffer(inst, vmfb_mm.map_memory()))
    vm = ireert.load_vm_modules(*modules, config=cfg)[-1]

    np_inp = [t.detach().cpu().numpy() for t in inputs]
    res = vm.main(*np_inp)
    return res.to_host() if hasattr(res, "to_host") else res


# ─────────────────────────────────────────────────────────────────────────────
# MLIR dump util
# ─────────────────────────────────────────────────────────────────────────────
def compile_with_dumps(stablehlo_text: str, backend: str, dump_dir: Path):
    """Compile StableHLO → vmfb and dump diagnostics into *dump_dir*."""
    opts = CompilerOptions(
        input_type=InputType.STABLEHLO,
        output_format=OutputFormat.FLATBUFFER_BINARY,
        target_backends=[backend],
        extra_args=["--mlir-print-ir-after-all"],
        crash_reproducer_path=str(dump_dir / "crash.mlir"),
        output_file=str(dump_dir / "module.vmfb"),
        extended_diagnostics=True,
    )
    (dump_dir / "stablehlo_input.mlir").write_text(stablehlo_text)

    try:
        return compile_str(stablehlo_text, **dataclasses.asdict(opts))
    except subprocess.CalledProcessError as cpe:
        (dump_dir / "compile_error.txt").write_text(
            cpe.stderr.decode(errors="ignore") if cpe.stderr else str(cpe)
        )
        raise
    except Exception:
        (dump_dir / "compile_error.txt").write_text(traceback.format_exc())
        raise


# ─────────────────────────────────────────────────────────────────────────────
# ExportOutput → StableHLO text
# ─────────────────────────────────────────────────────────────────────────────
def get_stablehlo_text(export_out) -> str:
    """Robustly fetch the StableHLO IR text from an `ExportOutput`."""
    # Newer turbine exposes a dict-like interface.
    if hasattr(export_out, "compiler_ir"):
        mod = export_out.compiler_ir("stablehlo")
        return str(mod)
    # Fallback: common private attribute `stablehlo`
    if hasattr(export_out, "stablehlo"):
        return str(export_out.stablehlo)
    # Last resort: look for first MLIR module
    if hasattr(export_out, "mlir_modules"):
        for m in export_out.mlir_modules.values():
            if "stablehlo" in m.name:
                return str(m)
    raise RuntimeError("Cannot extract StableHLO text from ExportOutput")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="*", help="block keys; omit = all")
    p.add_argument("--backend", default="llvm-cpu",
                   choices=["llvm-cpu", "cuda", "vulkan", "rocm"])
    p.add_argument("--mode", default="aot", choices=["aot", "jit"],
                   help="aot: external weights | jit: embed")
    p.add_argument("--csv", action="store_true", help="append latency csv")
    args = p.parse_args()

    models = args.model or ALL_MODELS
    results_root = Path("results/iree"); results_root.mkdir(exist_ok=True)
    compile_root = Path("results/compile_iree"); compile_root.mkdir(exist_ok=True)

    rows, ts = [], datetime.now().isoformat(timespec="seconds")

    for name in models:
        try:
            print(f"[{name}] export → compile → run …")
            model, dummy = load_model(name); model.eval()
            inputs = make_inputs(dummy)

            if args.mode == "aot":
                exported, weights = export_and_externalize(model, inputs)
            else:                                  # --mode jit
                exported, weights = aot.export(model, *inputs), {}

            # print(f"exported: {exported}")
            # print(f"weights: {weights}")

            mdl_out = results_root / name; mdl_out.mkdir(exist_ok=True)
            cmp_out = compile_root / name / args.backend
            cmp_out.mkdir(parents=True, exist_ok=True)

            if weights:
                save_weights(weights, mdl_out / "params.safetensors")
            np.save(mdl_out / "input.npy", inputs[0].detach().cpu().numpy())

            # StableHLO text 추출 후 저장
            shlo_text = get_stablehlo_text(exported)
            (cmp_out / "stablehlo.mlir").write_text(shlo_text)

            # compile & dumps
            vmfb_bytes = compile_with_dumps(shlo_text, args.backend, cmp_out)
            vmfb_mm = rt.mmap_vmfb(vmfb_bytes)

            # execute once
            t0 = time.perf_counter()
            run_vmfb(vmfb_mm, args.backend, inputs, weights)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            (mdl_out / "latency.txt").write_text(f"{latency_ms:.3f} ms\n")
            print(f"  ✓ {latency_ms:.3f} ms on {args.backend}")
            rows.append([ts, name, args.backend, f"{latency_ms:.3f}"])

        except Exception as e:
            reason = str(e).splitlines()[0]
            cmp_out.mkdir(parents=True, exist_ok=True)
            (cmp_out / "error_trace.txt").write_text(traceback.format_exc())
            print(f"  [ERROR] {name}: {reason}")
            rows.append([ts, name, args.backend, f"error: {reason}"])

    if args.csv:
        tag = "all" if not args.model else "_".join(args.model)
        csv_p = results_root / f"iree_latency_{tag}_{args.backend}.csv"
        with csv_p.open("a", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"[✓] CSV → {csv_p}")


if __name__ == "__main__":
    os.environ.setdefault("PJRT_DEVICE", "")  # silence XLA PJRT warnings
    main()

