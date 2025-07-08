#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IREE vmfb 실행 + 순수 커널 시간 측정 스크립트
Python 3.10
"""
import argparse, pathlib, time, statistics, csv, datetime, importlib
import numpy as np
import safetensors.numpy as stnp
import iree.runtime as ireert

from typing import List

import sys
# sys.path.append(str(pathlib.Path('models').resolve().parent))

ROOT_DIR    = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR   = ROOT_DIR / "models"

RESULTS_DIR = ROOT_DIR / "results" / "ireeflow"
PARAM_DIR   = RESULTS_DIR / "params"
INPUT_DIR   = RESULTS_DIR / "inputs"
VMFB_DIR    = RESULTS_DIR / "vmfb"

sys.path.append(str(ROOT_DIR))
# ──────────────────────────────────────────────────────────────────────────────
# Util
# ──────────────────────────────────────────────────────────────────────────────
# def discover():
#     """output/params 하위에서 모델 이름 자동 검색"""
#     return sorted(p.stem.replace(".safetensors", "")
#                   for p in OUT_PARAMS.glob("*_block.safetensors"))

def available_models(driver: str) -> List[str]:
    pattern = ""

    if driver == "local-sync" or driver == "local-task":
        pattern = "*-llvm-cpu.vmfb"
    elif driver == "cuda":
        pattern = "*-cuda.vmfb"

    return sorted(p.stem.rsplit("-", 1)[0] 
        for p in VMFB_DIR.glob(pattern))

# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────
def run_one(name: str, driver: str, warmup: int, iters: int):
    if driver == "cuda": vmfb_name = f"{name}-cuda" 
    else: vmfb_name = f"{name}-llvm-cpu"

    reqs = [PARAM_DIR / f"{name}.safetensors",
            VMFB_DIR   / f"{vmfb_name}.vmfb",
            INPUT_DIR  / f"{name}.npz"]
    if not all(f.exists() for f in reqs):
        print(f"[{name}] Files are not found → {reqs}")
        return

    # 1) Init IREE runtime
    config = ireert.Config(driver_name=driver)
    instance = config.vm_instance

    # 2) Parameter indexing (safetensors → const buffer)
    idx = ireert.ParameterIndex()
    for k, v in stnp.load_file(reqs[0]).items():
        idx.add_buffer(k, v.tobytes())


    # 3) Load vm_modules
    param_module = ireert.create_io_parameters_module(
        instance, idx.create_provider(scope="model")
    )
    hal_module = ireert.create_hal_module(instance, config.device)
    vm_module = ireert.VmModule.mmap(instance, str(reqs[1]))

    vm_modules = ireert.load_vm_modules(
        param_module, hal_module, vm_module,
        config=config,
    )
    main = vm_modules[-1].main

    # 4) Load inputs
    npz  = np.load(reqs[2])
    args = [npz[k] for k in sorted(npz.files)]

    # 5) warmup
    for _ in range(warmup):
        main(*args)

    # 6) Measure execution time
    dur_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _  = main(*args)
        dur_ms.append((time.perf_counter() - t0) * 1e3)

    mean_ms   = statistics.mean(dur_ms)
    median_ms = statistics.median(dur_ms)
    min_ms    = min(dur_ms)
    max_ms    = max(dur_ms)

    print(f"▶ {name:<20} | "
          f"mean {mean_ms:7.3f} ms | "
          f"median {median_ms:7.3f} ms | "
          f"min {min_ms:7.3f} ms | N={iters}")

    # 7) CSV 저장 --------------------------------------------------------------
    out_csv = RESULTS_DIR / "ireeflow.csv"
    with out_csv.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            name, driver, warmup, iters,
            f"{mean_ms:.3f}", f"{median_ms:.3f}",
            f"{min_ms:.3f}", f"{max_ms:.3f}"
        ])

# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="IREE vmfb 실행 & 커널 시간 측정"
    )
    ap.add_argument("model_name", nargs="?",
                    help="측정할 모델 이름(생략 시 output/params/* 자동 검색)")
    ap.add_argument("--driver", default="local-sync",
                    help="IREE 드라이버 (예: local-sync, local-task, cuda)")
    ap.add_argument("--warmup", type=int, default=5,
                    help="워밍업 실행 횟수")
    ap.add_argument("--iters", type=int, default=10,
                    help="측정 반복 횟수")
    args = ap.parse_args()

    names = [args.model_name] if args.model_name else available_models(args.driver)
    for n in names:
        run_one(n, args.driver, args.warmup, args.iters)


