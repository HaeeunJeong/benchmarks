#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_bench.py – PyTorch model loader + dummy run timer + (옵션) 메모리 트래커.
Python 3.11
"""
from __future__ import annotations

import argparse, csv, importlib, os, time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import psutil

# ─────────────────────────────────────────────────────────────────────────────
# (선택) 그래프
# ─────────────────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None  # --plot 옵션 사용 시만 필요

# ─────────────────────────────────────────────────────────────────────────────
# 모델 key 목록  ➜  models/<name>_block.py 와 1:1 대응
# ─────────────────────────────────────────────────────────────────────────────
VISION_KEYS = ["conv", "resnet", "mobilenet", "vit"]
GNN_KEYS    = ["gcn", "graphsage", "gat", "gatv2"]
LLM_KEYS    = ["bert", "gpt2", "llama", "deepseek"]
ALL_MODELS  = VISION_KEYS + GNN_KEYS + LLM_KEYS

# ─────────────────────────────────────────────────────────────────────────────
# 메모리 트래커 (백그라운드)
# ─────────────────────────────────────────────────────────────────────────────
def track_memory(pid: int, stop: list[bool], interval: float = 0.05):
    """RSS(MB) 시계열을 기록 – stop[0] 이 True 가 되면 종료."""
    proc = psutil.Process(pid)
    log: list[tuple[float, float]] = []
    t0 = time.perf_counter()
    while not stop[0]:
        rss = proc.memory_info().rss / (1024 ** 2)
        log.append((time.perf_counter() - t0, rss))
        time.sleep(interval)
    return log

# ─────────────────────────────────────────────────────────────────────────────
# 모델/더미 입력 로더 – **block module** 전용
# ─────────────────────────────────────────────────────────────────────────────
def load_model(name: str, device: str) -> tuple[torch.nn.Module, Any]:
    """
    models/<name>_block.py 를 import 후
        get_model()        -> nn.Module
        get_dummy_input()  -> (Tensor | tuple | shape-tuple)
    반환.
    """
    mod = importlib.import_module(f"models.{name.lower()}_block")
    model = mod.get_model().to(device)
    dummy = mod.get_dummy_input()
    return model, dummy

# ─────────────────────────────────────────────────────────────────────────────
# 추론 1-step 시간(ms) 측정
# ─────────────────────────────────────────────────────────────────────────────
def time_forward(model: torch.nn.Module, dummy: Any, device: str) -> float:
    if isinstance(dummy, tuple):
        # (1) 모든 원소가 Tensor → positional 호출
        if all(isinstance(t, torch.Tensor) for t in dummy):
            tensors = [t.to(device) for t in dummy]
            fn = lambda: model(*tensors)
        # (2) Vision 등 shape-tuple → 난수 Tensor 생성
        else:
            x = torch.randn(*dummy, device=device)
            fn = lambda: model(x)
    else:  # 단일 Tensor
        fn = lambda: model(dummy.to(device))

    # Warm-up
    for _ in range(5):
        with torch.no_grad(): fn()
    if device.startswith("cuda"): torch.cuda.synchronize()

    # Timed run
    start = time.perf_counter()
    with torch.no_grad(): fn()
    if device.startswith("cuda"): torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000.0  # → ms

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="*", help=f"model name ({'|'.join(ALL_MODELS)})")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--track-mem", action="store_true", help="RSS 추적")
    ap.add_argument("--plot", action="store_true", help="추적 시 그래프 저장")
    ap.add_argument("--csv", action="store_true", help="results/latency.csv 갱신")
    args = ap.parse_args()

    models = args.model or ALL_MODELS
    timestamp = datetime.now().isoformat(timespec="seconds")
    results: list[tuple[str, str, str, str, str]] = []

    for name in models:
        try:
            mdl, dummy = load_model(name, args.device)

            # ── 메모리 추적 스레드 시작
            stop = [False]; mem_log = []
            if args.track_mem:
                from threading import Thread
                t = Thread(target=lambda: mem_log.extend(track_memory(os.getpid(), stop)), daemon=True)
                t.start()

            ms = time_forward(mdl, dummy, args.device)

            # ── 메모리 추적 종료
            peak_mb = "NA"
            if args.track_mem:
                stop[0] = True; t.join()
                peak_mb = f"{max(r[1] for r in mem_log):.1f}"

            print(f"{name:10s}| {args.device} | {ms:8.2f} ms | peak {peak_mb} MB")
            results.append((timestamp, name, args.device, f"{ms:.2f}", peak_mb))

            # ── 그래프 저장
            if args.track_mem and args.plot and plt is not None:
                tt, mb = zip(*mem_log)
                plt.figure(figsize=(6, 3))
                plt.plot(tt, mb)
                plt.xlabel("time (s)"); plt.ylabel("RSS (MB)")
                plt.title(f"{name} CPU Memory Usage")
                out_png = Path(f"results/mem_{name}.png"); out_png.parent.mkdir(exist_ok=True)
                plt.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close()
                print(f"[✓] 그래프 저장: {out_png}")

        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            results.append((timestamp, name, args.device, f"ERROR: {e}", "NA"))

    # ── CSV
    if args.csv:
        out = Path("results/latency.csv"); out.parent.mkdir(exist_ok=True)
        with out.open("a", newline="") as f:
            csv.writer(f).writerows(results)
        print(f"[✓] Results saved to {out}")

if __name__ == "__main__":
    main()

