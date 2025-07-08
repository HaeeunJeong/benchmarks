#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
* 파라미터는 `.safetensors` 로 저장하고, IR에는 `dense_resource` 참조만 남김.
* vmfb 실행 시 `iree.runtime.ParameterIndex` 로 메모리 상에 가중치를 주입.
* 결과 artefacts (예: conv):
    results/ireeflow
      ├─ module.vmfb            (compiled binary)
      ├─ params.safetensors     (weights)
      ├─ input.npy              (샘플 입력, 재현용)
      └─ latency.txt            (1-shot ms)
"""
from __future__ import annotations

import argparse, csv, os, time, sys, shutil, subprocess, importlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import numpy as np
from safetensors.torch import save_file

import iree.turbine.aot as aot
# import iree.runtime as ireert
# import iree.runtime as rt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.run_bench import load_model, ALL_MODELS

# ---------------------------------------------------------------------------
# configuration
ROOT_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
UTIL_DIR  = ROOT_DIR / "utils"

RESULTS_DIR = ROOT_DIR    / "results" / "ireeflow"
PARAM_DIR   = RESULTS_DIR / "params"
INPUT_DIR   = RESULTS_DIR / "inputs"
MLIR_DIR    = RESULTS_DIR / "mlir"
DUMP_DIR    = RESULTS_DIR / "dump"
PATCH_DIR   = RESULTS_DIR / "patches"
VMFB_DIR    = RESULTS_DIR / "vmfb"
LOG_DIR     = RESULTS_DIR / "logs"

sys.path.append(str(ROOT_DIR))

# -------------------------------------------------------------------------------
# helpers
def _empty_dir(model: str) -> None:
    for d in (RESULTS_DIR, PARAM_DIR, INPUT_DIR, MLIR_DIR,
              DUMP_DIR, PATCH_DIR, LOG_DIR):
        d.mkdir(exist_ok=True, parents=True)
    for d in (MLIR_DIR, DUMP_DIR, PATCH_DIR, LOG_DIR):
        (d / model).mkdir(exist_ok=True, parents=True)

def _run(cmd: List[str], log: Path) -> None:
    if not log.exists():
        log.parent.mkdir(exist_ok=True, parents=True)
    res = subprocess.run(cmd, text=True, capture_output=True)
    merged = res.stdout + res.stderr
    log.write_text(merged, encoding="utf-8")
    if res.returncode:
        model_name = log.parent.name
        err_path = (LOG_DIR / model_name
                    / f"{log.stem}_error.log")
        err_path.write_text(merged, encoding="utf-8")
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"• Dump log: {log}\n"
            f"• Error log: {err_path}\n")

def _tensorize(d):  # tuple->rand tensor
    if isinstance(d, tuple) and all(isinstance(x, int) for x in d):
        return torch.randn(d)
    return d

def _flatten(x):
    return sum((_flatten(e) if isinstance(x, (tuple, list))
               else [x] for e in (x if isinstance(x, (tuple, list))
               else [x])), [])

# ---------------------------------------------------------------------------
# IREE STAGES
STAGES: List[Tuple[str, str, str, bool]] = [
    ("00.start", "start", "input", True),
    ("01.to-input", "start", "input", True),
    ("02.to-abi", "input", "abi", True),
    ("03.to-preproc", "abi", "preprocessing", True),
    ("04.to-gopt", "preprocessing", "global-optimization", False),
    ("05.to-dispatch", "global-optimization", "dispatch-creation", True),
    ("06.to-flow", "dispatch-creation", "flow", True),
    ("07.to-stream", "flow", "stream", True),
    ("08.to-exesrc", "stream", "executable-sources", True),
    ("09.to-execonfig", "executable-sources", "executable-configurations", True),
    ("10.to-exetar", "executable-configurations", "executable-targets", True),
    ("11.to-hal", "executable-targets", "hal", True),
    ("12.to-vm", "hal", "vm", True),
    ("13.to-end", "vm", "end", True),
]

# ---------------------------------------------------------------------------
# COMPILE STAGES
def _compile_stage(stage: Tuple[str, str, str, bool],
                   device: str, cur: Path,
                   mlir_dir: Path, dump_log: Path) -> Path:
    sid, c_from, c_to, _ = stage
    out = mlir_dir / f"{sid}{'.vmfb' if c_to == 'end' else '.mlir'}"
    dump_log.parent.mkdir(exist_ok=True, parents=True)

    extra_args = []
    if device == "llvm-cpu":
        extra_args = [
            "--iree-hal-target-backends=llvm-cpu",
            # "--iree-llvmcpu-target-cpu-features=host,+sse4.2,+avx2",
            "--iree-llvmcpu-target-cpu-features=host",
        ]
    elif device == "cuda":
        extra_args = [
            # "--iree-llvmcpu-target-cpu-features=host",
            "--iree-hal-target-backends=cuda",
            "--iree-cuda-target=rtx4090",
            "--iree-cuda-use-ptxas",
            # "--iree-cuda-target-features=+ptxas,+cuda-graph",
            "--iree-opt-const-expr-hoisting",
            # "--iree-opt-export-parameters=./params.safetensors",
            "--disable-constant-hoisting",
        ]

    DEBUG, OPT = 1, 1
    if DEBUG:
        extra_args.append("--mlir-print-ir-after-all")
    if OPT:
        extra_args.append("--iree-opt-level=O3")
        extra_args.append("--iree-dispatch-creation-opt-level=O3")
        extra_args.append("--iree-global-optimization-opt-level=O3")

    model_block = cur.stem
    mname = model_block.rsplit("_", 1)[0]

    if sid == "00.start":
        mod = importlib.import_module(f"models.{mname}_block")
        model = mod.get_model().eval()

        # 1) params
        save_file({k: v.detach().contiguous()
                   for k, v in model.state_dict().items()},
                  PARAM_DIR / f"{mname}.safetensors")
        # 2) torch mlir
        aot.externalize_module_parameters(model)
        dummy = _tensorize(mod.get_dummy_input())
        tensors = _flatten(dummy)
        t_mlir = mlir_dir / f"{mname}_torch.mlir"
        aot.export(model, *tensors).save_mlir(t_mlir)
        # 3) inputs
        np.savez(INPUT_DIR / f"{mname}.npz",
                 **{f"arg{i}": t.cpu().numpy()
                    for i, t in enumerate(tensors)})
        # 4) compile
        _run([
            "iree-compile",
            # "--iree-torch-decompose-complex-ops",
            # "--iree-torch-use-strict-symbolic-shapes",
            f"--compile-from={c_from}",
            f"--compile-to={c_to}",
            *extra_args, "-o", str(out), str(t_mlir)
        ], dump_log)
    else:
        _run([
            "iree-compile",
            f"--compile-from={c_from}",
            f"--compile-to={c_to}",
            *extra_args, "-o", str(out), str(cur)
        ], dump_log)
    return out

def _diff(stage: Tuple[str, str, str, bool],
          dump_dir: Path, patch_dir: Path) -> None:
    if not stage[3]:
        return
    sid = stage[0]
    subprocess.run([
        "python", str(UTIL_DIR / "passdiff.py"),
        str(dump_dir / f"{sid}_dump.mlir"),
        "--patch", str(patch_dir / f"{sid}.patch"),
        "--patch-only"
    ], check=True)

# ---------------------------------------------------------------------------
# PIPELINE
def build_model(model: str, device: str, run_test: bool) -> None:
    _empty_dir(model)
    mlir_dir, dump_dir = MLIR_DIR / model, DUMP_DIR / model
    patch_dir = PATCH_DIR / model

    print(f" (1/{len(STAGES)}) {STAGES[0][0]}")
    cur = _compile_stage(STAGES[0], device,
                         MODEL_DIR / f"{model}.py",
                         mlir_dir, dump_dir / f"{model}_dump.mlir")
    for idx, st in enumerate(STAGES[1:], 2):
        print(f" ({idx}/{len(STAGES)}) {st[0]}")
        cur = _compile_stage(
            st, device, cur, mlir_dir,
            dump_dir / f"{st[0]}_dump.mlir")
        _diff(st, dump_dir, patch_dir)

    VMFB_DIR.mkdir(exist_ok=True, parents=True)
    vmfb = VMFB_DIR / f"{model}-{device}.vmfb"
    shutil.copy2(mlir_dir / "13.to-end.vmfb", vmfb)
    print(f"[build] VMFB saved → {vmfb}")

    if run_test:
        subprocess.run([
            "python", str(UTIL_DIR / "runvmfb.py"),
            model, "--driver",
            ("cuda" if device == "cuda" else "local-sync")
        ], check=True)

def infer_model(model: str, device: str) -> None:
    vmfb = VMFB_DIR / f"{model}-{device}.vmfb"
    if not vmfb.exists():
        raise FileNotFoundError(vmfb)
    driver = "cuda" if device == "cuda" else "local-sync"
    subprocess.run([
        "python", str(UTIL_DIR / "runvmfb.py"),
        model, "--driver", driver
    ], check=True)

# ---------------------------------------------------------------------------
# CLI
def available_models() -> List[str]:
    return sorted(p.stem.rsplit("_", 1)[0] for p in MODEL_DIR.glob("*_block.py"))

def main() -> None:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Compile models to IREE vmfb / run vmfb")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    # build
    pb = sub.add_parser("build", help="Compile models to IREE vmfb")
    g1 = pb.add_mutually_exclusive_group(required=False)
    g1.add_argument("--model", nargs="?", metavar="NAME",
                    help="Single model; omit=all")
    pb.add_argument("--device", default="cuda",
                    help="cuda | llvm-cpu (default: cuda)")
    pb.add_argument("--run-test", action="store_true",
                    help="Run test after compilation")

    # infer
    pr = sub.add_parser("infer", help="Run IREE vmfb")
    g2 = pr.add_mutually_exclusive_group(required=False)
    g2.add_argument("--model", nargs="?", metavar="NAME",
                    help="Single model; omit=all")
    pr.add_argument("--device", default="cuda",
                    help="cuda | llvm-cpu (default: cuda)")

    if len(sys.argv) == 1:
        p.print_help(); sys.exit(0)
    args = p.parse_args()

    if args.command == "build":
        targets = available_models() if args.model is None else [args.model]
        for i, m in enumerate(targets, 1):
            print(f"[{i}/{len(targets)}] build {m} on {args.device} …")
            try:
                build_model(m, args.device, args.run_test)
            except Exception as e:
                print(f"[ERR] build {m} failed: {e}", file=sys.stderr)

    elif args.command == "infer":
        targets = available_models() if args.model is None else [args.model]
        for i, m in enumerate(targets, 1):
            print(f"[{i}/{len(targets)}] infer → {m} on {args.device} …")
            try:
                infer_model(m, args.device)
            except Exception as e:
                print(f"[ERR] infer {m} failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

