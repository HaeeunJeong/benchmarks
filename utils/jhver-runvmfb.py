#!/usr/bin/env python3
"""
runvmfb.py – Compare IREE VMFB vs native PyTorch inference
==========================================================
* Loads parameters from __stablehlo_dir (trimming head-padding if present).
* Runs the compiled VM module **and** the original Python model on the same
  inputs, printing execution time and output-difference statistics.
"""
from __future__ import annotations

import argparse, importlib, json, time, sys, pathlib
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import iree.runtime as rt
import numpy as np
import torch

# models/ 디렉터리 경로 추가
sys.path.append(str(pathlib.Path('models').resolve().parent))

# ----------------------------------------------------------------------------- #
# dtype map
_DTYPE_TABLE = {
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": np.float16,
    "int64": np.int64,
    "int32": np.int32,
    "int8": np.int8,
    "uint8": np.uint8,
    "bool": np.bool_,
}
def _np_dtype(name: str) -> np.dtype:
    try:
        return _DTYPE_TABLE[name]
    except KeyError:
        raise ValueError(f"Unsupported StableHLO dtype: {name}") from None

# ----------------------------------------------------------------------------- #
# 입력/파라미터 유틸
def _load_param(root: Path, blob: str, sig: dict) -> np.ndarray:
    for sub in ("data", "constants"):
        f = root / sub / blob
        if f.exists():
            arr = np.fromfile(f, dtype=_np_dtype(sig["dtype"]))
            need = int(np.prod(sig["shape"]))
            if arr.size < need:
                raise ValueError(f"{blob}: file too small ({arr.size} < {need})")
            if arr.size > need:
                pad = arr.size - need
                print(
                    f"[warn] {blob}: detected {pad} extra elements; "
                    f"keeping LAST {need} values (head-padding)"
                )
                arr = arr[-need:]
            return arr.reshape(sig["shape"])
    raise FileNotFoundError(f"Parameter blob '{blob}' not found")

def _shape_ok(sig_shape: Sequence[int], arr_shape: Tuple[int, ...]) -> bool:
    return len(sig_shape) == len(arr_shape) and all(
        s == 0 or s == a for s, a in zip(sig_shape, arr_shape)
    )

def _torch_to_np(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    raise TypeError(f"Unsupported tensor type: {type(x)}")

# ---------------- 런타임 입력 정규화 ---------------- #
def _to_tensor_runtime(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, (int, float)):
        return torch.tensor(x)
    if isinstance(x, (tuple, list)) and all(isinstance(i, int) for i in x):
        return torch.randn(*x)
    raise TypeError(f"Unsupported element in dummy_input: {type(x)}")

def _normalize_runtime_inputs(raw: Any) -> List[torch.Tensor]:
    """
    get_dummy_input() 반환값을 torch.Tensor 리스트로 변환.
    """
    # if torch.is_tensor(raw) or isinstance(raw, (int, float, tuple, list)):
    #     raw = (raw,) if not isinstance(raw, (tuple, list)) else raw
    #     return [_to_tensor_runtime(e) for e in raw]
    # raise TypeError(f"Unsupported dummy_input type: {type(raw)}")
        # ① shape-tuple(list) → 단일 Tensor
    if isinstance(raw, (tuple, list)) and all(isinstance(i, int) for i in raw):
        return [torch.randn(*raw)]
    # ② 스칼라 / 단일 Tensor
    if torch.is_tensor(raw) or isinstance(raw, (int, float)):
        return [_to_tensor_runtime(raw)]
    # ③ 여러 Tensor / shape-tuple 이 섞인 시퀀스
    if isinstance(raw, (tuple, list)):
        return [_to_tensor_runtime(e) for e in raw]
    raise TypeError(f"Unsupported dummy_input type: {type(raw)}")
# --------------------------------------------------- #

def _assemble_inputs(
    meta: dict, root: Path, runtime: Sequence[np.ndarray]
) -> List[np.ndarray]:
    ordered: List[np.ndarray] = []
    rt_idx = 0
    sigs = meta["input_signature"]
    for idx, loc in enumerate(meta["input_locations"]):
        if loc["type_"] == "parameter":
            ordered.append(_load_param(root, loc["name"], sigs[idx]))
        elif loc["type_"] == "input_arg":
            if rt_idx >= len(runtime):
                raise ValueError("get_dummy_input() provided too few tensors")
            t = runtime[rt_idx]
            rt_idx += 1
            sig_shape = tuple(sigs[idx]["shape"])
            # shape 맞추기 시도
            if not _shape_ok(sig_shape, t.shape):
                # (1) 요소 수 동일 → reshape
                if np.prod(sig_shape) == np.prod(t.shape):
                    t = t.reshape(sig_shape)
                # (2) 4-D NCHW↔NHWC 전치
                elif len(sig_shape) == len(t.shape) == 4 and sig_shape == t.shape[::-1]:
                    t = t.transpose(0, 3, 1, 2)  # NHWC→NCHW or 반대
                else:
                    raise ValueError(
                        f"Runtime tensor shape {t.shape} incompatible with {sig_shape}"
                    )
            if t.dtype != _np_dtype(sigs[idx]["dtype"]):
                t = t.astype(_np_dtype(sigs[idx]["dtype"]))
            ordered.append(t)
        else:
            raise ValueError(f"Unknown input location type: {loc['type_']}")
    if rt_idx != len(runtime):
        raise ValueError("get_dummy_input() returned extra tensors not expected")
    return ordered

def _load_module(vmfb: Path, device: str):
    cfg = rt.Config(device)
    ctx = rt.SystemContext(config=cfg)
    vm_mod = rt.VmModule.mmap(ctx.instance, str(vmfb))
    ctx.add_vm_module(vm_mod)
    return ctx.modules.__getattr__(vm_mod.name)

# ----------------------------------------------------------------------------- #
# Main
def main() -> None:
    argp = argparse.ArgumentParser(
        description="Compare IREE VMFB vs PyTorch outputs and timing"
    )
    argp.add_argument("--model", required=True, help="model name (without .py)")
    argp.add_argument("--device", default="cpu", help="cpu|cuda|vulkan|llvm")
    args = argp.parse_args()

    vmfb = Path("__output_vmfb") / f"{args.model}.vmfb"
    meta_f = Path("__stablehlo_dir") / args.model / "functions" / "forward.meta"
    root = Path("__stablehlo_dir") / args.model
    if not vmfb.exists() or not meta_f.exists():
        raise FileNotFoundError("Missing VMFB or metadata for model")

    # -------- PyTorch 모델 & 입력 --------
    mdl_mod = importlib.import_module(f"models.{args.model}")
    pyt_model = mdl_mod.get_model().eval()
    raw_inputs = mdl_mod.get_dummy_input()
    torch_inputs = _normalize_runtime_inputs(raw_inputs)
    runtime_np = [_torch_to_np(t) for t in torch_inputs]

    # -------- IREE 입력 조립 --------
    meta = json.loads(meta_f.read_text())
    vm_inputs = _assemble_inputs(meta, root, runtime_np)

    # -------- IREE 추론 --------
    vm_mod = _load_module(vmfb, device=args.device)
    entry = next(
        n for n in (meta.get("name", ""), "forward", "main") if n and hasattr(vm_mod, n)
    )
    t0 = time.perf_counter()
    vm_outs = getattr(vm_mod, entry)(*vm_inputs)
    vm_elapsed = time.perf_counter() - t0
    vm_outs = vm_outs if isinstance(vm_outs, tuple) else (vm_outs,)

    # -------- PyTorch 추론 --------
    with torch.no_grad():
        t1 = time.perf_counter()
        torch_outs = pyt_model(*torch_inputs)
        torch_elapsed = time.perf_counter() - t1
    torch_outs = (
        torch_outs if isinstance(torch_outs, (tuple, list)) else (torch_outs,)
    )
    torch_outs_np = [_torch_to_np(t) for t in torch_outs]

    # -------- 결과 비교 --------
    print("Inference results")
    print(f"  IREE {args.device:>5}: {vm_elapsed*1000:.3f} ms")
    print(f"  PyTorch CPU : {torch_elapsed*1000:.3f} ms")
    for i, (v, t) in enumerate(zip(vm_outs, torch_outs_np)):
        diff = np.abs(v - t)
        print(
            f"  out{i}: shape={v.shape}, max_abs_diff={diff.max():.6e}, "
            f"mean_abs_diff={diff.mean():.6e}"
        )

if __name__ == "__main__":
    main()

