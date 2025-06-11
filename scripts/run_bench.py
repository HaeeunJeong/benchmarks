#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_bench.py – PyTorch model loader + dummy run timer.
Python 3.11
"""

from __future__ import annotations

import argparse
import time
import importlib
import csv

from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

import torch

# Define common input shapes
LLM_SEQ_LEN: int = 32
VISION_INPUT_SHAPE: Tuple[int, int, int, int] = (1, 3, 224, 224)  # (batch_size, channels, height, width)

HF_MODELS: Dict[str, str] = {
    "bert":               "bert-base-uncased",
    "gpt2":               "gpt2-xl",
    # "llama":              "meta-llama/Meta-Llama-3-8B",
    "llama":            "meta-llama/Llama-3.2-3B_INT4_EO8",
    "llama-quant":      "meta-llama/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8",
    "llama-qlora":      "meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8",
    "deepseek":         "deepseek-ai/deepseek-R1-Distill-Qwen-1.5B",
}

TV_MODELS: Dict[str, Tuple[str, str, int]] = {
    "resnet":     ("resnet18", "ResNet18", 224),
    "mobilenet":  ("mobilenet_v3_small", "MobileNet_V3_Small", 224),
    "vit":        ("vit_b_16", "ViT_B_16", 224),
}

ALL_MODELS = ["conv", "gat"] + list(TV_MODELS.keys()) + list(HF_MODELS.keys())

# Model Loader
def load_model(name: str):
    name = name.lower()

    if name == "conv": # Custom convolutional block
        mod = importlib.import_module("models.conv_block")
        return mod.ConvConvReLU(), VISION_INPUT_SHAPE

    if name == "gat": # PyG (PyTorch Geometric)
        mod = importlib.import_module("models.gat_block")
        model = mod.GATBlock()
        dummy_input = mod.get_dummy_input()
        return model, dummy_input

    if name in TV_MODELS: # torchvision models
        fn_name, weight_name, _ = TV_MODELS[name]
        tv = importlib.import_module("torchvision.models")
        fn = getattr(tv, fn_name)
        weights_enum_name = weight_name + "_Weights"
        try:
            weights_enum = getattr(tv, weights_enum_name)
            model = fn(weights=weights_enum.IMAGENET1K_V1)
        except AttributeError:
            # Fallback for older torchvision versions without weights enum
            model = fn(pretrained=True)

        return model, VISION_INPUT_SHAPE

    if name in HF_MODELS: # Hugging Face models
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(HF_MODELS[name])
        mdl = AutoModel.from_pretrained(HF_MODELS[name])

        vocab = getattr(tok, "vocab_size", 50257)  # Default for GPT-2
        dummy_ids = torch.randint(0, vocab, (1, LLM_SEQ_LEN), dtype=torch.long)
        dummy = {"input_ids": dummy_ids, "attention_mask": torch.ones_like(dummy_ids)}

        return mdl, dummy

    raise ValueError(f"Unknown model key: {name!r}")

def time_forward(model: torch.nn.Module, dummy_input: Any, device: str ="cuda") -> float:
    model = model.to(device)

    if isinstance(dummy_input, tuple):
        # Vision or GAT model
        if len(dummy_input) == 2 and isinstance(dummy_input[0], torch.Tensor):
            x, edge_index = dummy_input
            x, edge_index = x.to(device), edge_index.to(device)
            fn = lambda: model(x, edge_index)
        else:
            x = torch.randn(*dummy_input, device=device)
            fn = lambda: model(x)
    else: # LLM Dictionary input
        dummy_input = {k: v.to(device) for k, v in dummy_input.items()}
        fn = lambda: model(**dummy_input)

    # ------- Warmup runs -------
    for _ in range(5):
        with torch.no_grad():
            fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # ------- Timing run -------
    start = time.perf_counter()
    with torch.no_grad():
        fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000.0  # ms

# CLI
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "model", 
        nargs="*", 
        help="Model(s) to run: conv | gat | resnet | mobilenet | vit | bert | gpt2 | llama | deepseek",
    )
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--csv", action="store_true", help="Save results to results/latency.csv")
    args = p.parse_args()

    models_to_run = args.model if args.model else ALL_MODELS
    results = []
    timestamp = datetime.now().isoformat(timespec='seconds')

    for model_name in models_to_run:
        try:
            mdl, dummy = load_model(model_name)
            ms = time_forward(mdl, dummy, args.device)
            print(f"{model_name:10s} | {args.device} | {ms:8.10f} ms")
            results.append((timestamp, model_name, args.device, f"{ms:.10f} ms"))
        except Exception as e:
            print(f"[ERROR] {model_name}: {e}")
            results.append((timestamp, model_name, args.device, str(e)))

    if args.csv:
        out_path = Path(f"results/latency_{model_name}.csv")
        out_path.parent.mkdir(exist_ok=True)
        with out_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(results)
        print(f"[✓] Results saved to {out_path}")

if __name__ == "__main__":
    main()

