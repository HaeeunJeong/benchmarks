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
from typing import Tuple, Dict, Any

import torch
import psutil

# ---------- 그래프 그리기용 (선택) ----------
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None  # --plot 옵션 사용 시 확인

# ---------- 공통 상수 ----------
LLM_SEQ_LEN = 32
VISION_INPUT_SHAPE = (1, 3, 224, 224)

HF_MODELS: Dict[str, str] = {
    "bert":         "bert-base-uncased",
    "gpt2":         "gpt2-xl",
    # "llama":      "meta-llama/Meta-Llama-3-8B",
    "llama":        "meta-llama/Llama-3.2-3B",
    "llama-quant":  "meta-llama/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8",
    "llama-qlora":  "meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8",
    "deepseek":     "deepseek-ai/deepseek-R1-Distill-Qwen-1.5B",
}
TV_MODELS: Dict[str, Tuple[str, str, int]] = {
    "resnet":    ("resnet18", "ResNet18", 224),
    "mobilenet": ("mobilenet_v3_small", "MobileNet_V3_Small", 224),
    "vit":       ("vit_b_16", "ViT_B_16", 224),
}
ALL_MODELS = ["conv", "gat"] + list(TV_MODELS.keys()) + list(HF_MODELS.keys())

# ---------- 메모리 트래커 ----------
def track_memory(pid: int, stop_flag: list[bool], interval: float = 0.05):
    """백그라운드에서 RSS(MB) 시계열을 기록하고 리스트를 리턴."""
    proc = psutil.Process(pid)
    mem_log: list[tuple[float, float]] = []
    t0 = time.perf_counter()
    while not stop_flag[0]:
        rss = proc.memory_info().rss / (1024**2)  # MB
        mem_log.append((time.perf_counter() - t0, rss))
        time.sleep(interval)
    return mem_log

# ---------- 모델 로딩 (지면 관계상 이전 답변의 dtype/quant 지원 버전 재사용) ----------

def load_model(
    name: str,
    device: str,
) -> Tuple[torch.nn.Module, Any]:
    name = name.lower()

    # (1) Conv 블록
    if name == "conv":
        mod = importlib.import_module("models.conv_block")
        model = mod.ConvConvReLU().to(device=device)
        return model, VISION_INPUT_SHAPE

    # (2) GAT 블록
    if name == "gat":
        mod = importlib.import_module("models.gat_block")
        model = mod.GATBlock().to(device=device)
        dummy_input = mod.get_dummy_input()  # (x, edge_index)
        return model, dummy_input

    # (3) TorchVision
    if name in TV_MODELS:
        fn_name, *_ = TV_MODELS[name]
        tv = importlib.import_module("torchvision.models")
        fn = getattr(tv, fn_name)
        model = fn(weights="DEFAULT" if "DEFAULT" in fn.__annotations__.get("weights", "") else None)
        model = model.to(device=device)
        return model, VISION_INPUT_SHAPE

    # (4) Hugging Face LLM
    if name in HF_MODELS:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from sentence_transformers import SentenceTransformer

        kwargs = {
            "device_map": "auto" if device == "cuda" else None,
        }

        tok = AutoTokenizer.from_pretrained(HF_MODELS[name], use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODELS[name], 
            **kwargs, 
            trust_remote_code=True)

        vocab = tok.vocab_size
        dummy_ids = torch.randint(0, vocab, (1, LLM_SEQ_LEN), device=device)
        dummy = {"input_ids": dummy_ids, "attention_mask": torch.ones_like(dummy_ids)}

        return model, dummy

    raise ValueError(f"Unknown model key: {name!r}")

# ---------- 추론 시간 측정 ----------
def time_forward(model: torch.nn.Module, dummy_input: Any, device: str) -> float:
    # 이미 올바른 device/dtype 으로 로드됨
    if isinstance(dummy_input, tuple):
        if len(dummy_input) == 2 and isinstance(dummy_input[0], torch.Tensor):
            x, edge_index = dummy_input
            x, edge_index = x.to(device), edge_index.to(device)
            fn = lambda: model(x, edge_index)
        else:
            x = torch.randn(*dummy_input, device=device)
            fn = lambda: model(x)
    else:
        dummy_input = {k: v.to(device) for k, v in dummy_input.items()}
        fn = lambda: model(**dummy_input)

    # 워밍업
    for _ in range(5):
        with torch.no_grad():
            fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000.0  # ms


# ---------- CLI ----------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="*", help="model name (conv, gat, resnet, mobilenet, vit, bert, gpt2, llama, llama-quant, llama-qlora, deepseek)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--track-mem", action="store_true", help="RSS 추적")
    p.add_argument("--plot", action="store_true", help="추적 시 그래프 png 저장/보기")
    p.add_argument("--csv", action="store_true")
    args = p.parse_args()

    models_to_run = args.model or ALL_MODELS
    timestamp = datetime.now().isoformat(timespec="seconds")
    results = []

    for m in models_to_run:
        try:
            mdl, dummy = load_model(m, args.device)

            # ===== 메모리 백그라운드 스레드 시작 =====
            stop_flag = [False]
            mem_log = []
            if args.track_mem:
                from threading import Thread
                th = Thread(target=lambda: mem_log.extend(track_memory(os.getpid(), stop_flag)), daemon=True)
                th.start()

            ms = time_forward(mdl, dummy, args.device)

            # ===== 메모리 트래커 종료 =====
            if args.track_mem:
                stop_flag[0] = True
                th.join()
                peak_mb = max(r[1] for r in mem_log)
            else:
                peak_mb = "NA"

            print(f"{m:10s} | {args.device}"
                  f"| {ms:8.2f} ms | peak {peak_mb} MB")
            results.append((timestamp, m, args.device, f"{ms:.2f}", peak_mb))

            # -------- 그래프 ----------
            if args.track_mem and args.plot:
                if plt is None:
                    print("[WARN] matplotlib 미설치 – 그래프 생략")
                else:
                    t, mb = zip(*mem_log)
                    plt.figure(figsize=(6, 3))
                    plt.plot(t, mb)
                    plt.xlabel("time (s)")
                    plt.ylabel("RSS (MB)")
                    plt.title(f"{m} CPU Memory Usage")
                    out_png = Path(f"results/mem_{m}.png")
                    out_png.parent.mkdir(exist_ok=True)
                    plt.savefig(out_png, dpi=120, bbox_inches="tight")
                    plt.close()
                    print(f"[✓] 그래프 저장: {out_png}")

        except Exception as e:
            print(f"[ERROR] {m}: {e}")
            results.append((timestamp, m, args.device, f"ERROR: {e}", "NA"))

    # CSV 저장
    if args.csv:
        out_path = Path("results/latency.csv")
        out_path.parent.mkdir(exist_ok=True)
        with out_path.open("a", newline="") as f:
            csv.writer(f).writerows(results)
        print(f"[✓] Results saved to {out_path}")

if __name__ == "__main__":
    main()

