#!/usr/bin/env python3
import argparse, pathlib, sys, iree.compiler.tools as irec
# sys.path.append(str(pathlib.Path('models').resolve().parent))

ROOT      = pathlib.Path(__file__).parent
print(f"ROOT: {ROOT}")
OUT_MLIR  = ROOT / "../../results/iree-output" / "mlir"
print(f"OUT_MLIR: {OUT_MLIR}")
OUT_VMFB  = ROOT / "../../results/iree-output" / "vmfb"
OUT_VMFB.mkdir(parents=True, exist_ok=True)

def discover():  # *_block.mlir
    return sorted(p.stem for p in OUT_MLIR.glob("*_block.mlir"))

def compile_one(name: str):
    mlir_f = OUT_MLIR / f"{name}.mlir"
    if not mlir_f.exists():
        print(f"[{name}] .mlir 없음")
        return
    try:
        vmfb = irec.compile_str(mlir_f.read_text(),
                            target_backends=["llvm-cpu"],
                            input_type="torch",
                            extra_args=[
                                "--iree-llvmcpu-target-cpu-features=host",
                                "--iree-torch-decompose-complex-ops",
                                "--iree-torch-use-strict-symbolic-shapes",
                                f"--iree-opt-level=O{1}"
                            ],
                            # output_format="vmfb"
                            )
        (OUT_VMFB / f"{name}.vmfb").write_bytes(vmfb)
        print(f"✓ [{name}] vmfb 저장")
    except Exception as e:
        print(f"[{name}] 컴파일 실패: {e}", file=sys.stderr)
        return

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_name", nargs="?")
    arg = ap.parse_args()
    for n in ([arg.model_name] if arg.model_name else discover()):
        compile_one(n)

