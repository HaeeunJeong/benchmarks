#!/usr/bin/env python3
"""
compile_mlir.py
────────────────
OUT_MLIR/ 에 있는 *.mlir 파일을 IREE VMFB 로 컴파일한다.

사용법
------
python compile_mlir.py                # 폴더 내 모든 모델, input-type = torch (기본)
python compile_mlir.py --dialect linalg
python compile_mlir.py bert_block     # 단일 모델
python compile_mlir.py resnet_block --dialect linalg
"""
from __future__ import annotations
import argparse, pathlib, sys, shutil
import iree.compiler.tools as irec

ROOT      = pathlib.Path(__file__).parent
OUT_MLIR  = ROOT / "../../results/iree-output/mlir"
OUT_VMFB  = ROOT / "../../results/iree-output/vmfb"
if OUT_VMFB.exists():
    shutil.rmtree(OUT_VMFB)
OUT_VMFB.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────
def _discover() -> list[str]:
    return sorted(p.stem for p in OUT_MLIR.glob("*_block.mlir"))

def _compile_one(name: str, dialect: str) -> None:
    mlir_f = OUT_MLIR / f"{name}.mlir"
    if not mlir_f.exists():
        print(f"[{name}] .mlir 파일 없음")
        return

    # input-type 매핑
    if dialect == "torch":
        in_type = "torch"
        extra = [
            "--iree-torch-decompose-complex-ops",
            # "--iree-torch-use-strict-symbolic-shapes",
        ]
    elif dialect == "linalg":
        in_type = "auto"
        extra = [
            "--compile-from=input",
            "--compile-to=end",
            # linalg-on-tensors → flow 로 갈 때 별도 옵션이 필요 없다. 필요 시 여기 추가 
        ]
    else:
        print(f"[{name}] Unsupported dialect: {dialect}")
        return

    try:
        vmfb_blob = irec.compile_str(
            mlir_f.read_text(),
            target_backends=["llvm-cpu"],
            input_type=in_type,          # torch or linalg
            extra_args=[
                "--iree-llvmcpu-target-cpu-features=host",
                # "--canonicalize", "--cse",
                *extra,
            ],
        )
        (OUT_VMFB / f"{name}.vmfb").write_bytes(vmfb_blob)
        print(f"✓ [{name}] vmfb 저장 ({dialect})")
    except Exception as e:
        print(f"[{name}] 컴파일 실패 ({dialect}): {e}", file=sys.stderr)

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_name", nargs="?",
                    help="models 파일 stem (default: ALL)")
    ap.add_argument("--dialect", choices=["torch", "linalg"],
                    default="linalg", help="input MLIR dialect (default: linalg)")
    args = ap.parse_args()

    targets = [args.model_name] if args.model_name else _discover()
    for t in targets: _compile_one(t, args.dialect)

