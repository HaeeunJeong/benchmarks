#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IREE vmfb 실행 + 순수 커널 시간 측정 스크립트
Python 3.10
"""
import argparse, pathlib, time, statistics, csv, datetime
import numpy as np
import safetensors.numpy as stnp
import iree.runtime as ireert

import sys
sys.path.append(str(pathlib.Path('models').resolve().parent))

ROOT        = pathlib.Path(__file__).parent
OUT_PARAMS  = ROOT / "../../results/iree-output" / "params"
OUT_INPUT   = ROOT / "../../results/iree-output" / "input"
OUT_VMFB    = ROOT / "../../results/iree-output" / "vmfb"
OUT_CSV     = ROOT / "../../results/iree-output" / "latency.csv"   # ← 결과 CSV

# ──────────────────────────────────────────────────────────────────────────────
# Util
# ──────────────────────────────────────────────────────────────────────────────
def discover():
    """output/params 하위에서 모델 이름 자동 검색"""
    return sorted(p.stem.replace(".safetensors", "")
                  for p in OUT_PARAMS.glob("*_block.safetensors"))

# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────
def run_one(name: str, driver: str, warmup: int, iters: int):
    reqs = [OUT_PARAMS / f"{name}.safetensors",
            OUT_VMFB   / f"{name}.vmfb",
            OUT_INPUT  / f"{name}.npz"]
    if not all(f.exists() for f in reqs):
        print(f"[{name}] 필요한 파일이 없습니다 → {reqs}")
        return

    # 1) 런타임 초기화
    cfg  = ireert.Config(driver)
    inst = cfg.vm_instance

    # 2) 파라미터 인덱스 작성 (safetensors → const buffer)
    pidx = ireert.ParameterIndex()
    for k, v in stnp.load_file(reqs[0]).items():
        pidx.add_buffer(k, v.tobytes())

    # 3) VM 모듈 묶음 생성
    modules = ireert.load_vm_modules(
        ireert.create_io_parameters_module(inst, pidx.create_provider("model")),
        ireert.create_hal_module(inst, cfg.device),
        ireert.VmModule.mmap(inst, str(reqs[1])),
        config=cfg
    )
    main = modules[-1].main

    # 4) 입력 로드
    npz  = np.load(reqs[2])
    args = [npz[k] for k in sorted(npz.files)]

    # 5) 워밍업
    for _ in range(warmup):
        main(*args)

    # 6) 순수 실행 시간 측정
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
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            name, driver, "CPU", warmup, iters,
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
    ap.add_argument("--driver", default="local-task",
                    help="IREE 드라이버 (예: local-sync, cuda-sync, vulkan-sync)")
    ap.add_argument("--warmup", type=int, default=5,
                    help="워밍업 실행 횟수")
    ap.add_argument("--iters", type=int, default=10,
                    help="측정 반복 횟수")
    args = ap.parse_args()

    names = [args.model_name] if args.model_name else discover()
    for n in names:
        run_one(n, args.driver, args.warmup, args.iters)

