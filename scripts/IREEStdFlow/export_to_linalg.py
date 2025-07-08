#!/usr/bin/env python3
"""
make_iree_linalg_mlir.py
────────────────────────
IREE만 사용해 models/<name>_block.py 를 linalg(+linalg_ext) MLIR로 변환.
* 성공-여부와 무관하게 모델 루프를 끝까지 돌며,
  컴파일 실패 시 <logs>/<model>.log 파일에 stdout+stderr 저장 후 continue.
"""
from __future__ import annotations
import argparse, importlib, shutil, subprocess, sys
from pathlib import Path
from typing import Any, List
import torch, numpy as np
from safetensors.torch import save_file
import iree.turbine.aot as aot

# ────────────────────────────────────────────────────────────────
sys.path.append(str(Path("models").resolve().parent))

ROOT        = Path(__file__).parent
MODELS_DIR  = ROOT / "../../models"
OUT_BASE    = ROOT / "../../results/iree-output"
OUT_PARAMS  = OUT_BASE / "params"
OUT_MLIR    = OUT_BASE / "mlir"
OUT_INPUT   = OUT_BASE / "input"
OUT_LOGS    = OUT_BASE / "logs"
TMP_DIR     = OUT_BASE / "__tmp"

def _mkdirs():
    for d in (OUT_PARAMS, OUT_MLIR, OUT_INPUT, OUT_LOGS, TMP_DIR):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

def _discover_models() -> List[str]:
    return sorted(p.stem for p in MODELS_DIR.glob("*_block.py"))

# ----- dummy-input helpers -------------------------------------
def _tensorize(x: Any) -> Any:
    if isinstance(x, tuple) and all(isinstance(i, int) for i in x):
        return torch.randn(*x)
    if isinstance(x, (list, tuple)):
        return type(x)(_tensorize(e) for e in x)
    return x

def _flatten(obj: Any) -> List[torch.Tensor]:
    if isinstance(obj, (list, tuple)):
        res: List[torch.Tensor] = []
        for o in obj:
            res.extend(_flatten(o))
        return res
    if torch.is_tensor(obj):
        return [obj]
    raise TypeError(type(obj))

# ----- subprocess helper ---------------------------------------
def _run(cmd: List[str], log: Path) -> bool:
    res = subprocess.run(cmd, text=True, capture_output=True)
    log.write_text(res.stdout + res.stderr, encoding="utf-8")
    if res.returncode != 0:
        print(f"[ERROR] {cmd[0]} failed → see {log.name}")
        return False
    return True
# ---------------------------------------------------------------

def process(model_name: str) -> None:
    try:
        mod   = importlib.import_module(f"models.{model_name}")
    except Exception as e:
        (OUT_LOGS / f"{model_name}_import.log").write_text(str(e))
        print(f"[SKIP] import 실패: {model_name}")
        return

    model = mod.get_model().eval()

    # 1) save params
    save_file({k: v.detach().cpu().contiguous()
               for k, v in model.state_dict().items()},
              OUT_PARAMS / f"{model_name}.safetensors")

    # 2-A) Torch-dialect IR via turbine.aot
    dummy  = _tensorize(mod.get_dummy_input())
    tensors = _flatten(dummy)
    aot.externalize_module_parameters(model)
    torch_mlir = TMP_DIR / f"{model_name}_torch.mlir"
    aot.export(model, *tensors).save_mlir(torch_mlir)

    # 2-B) IREE compile to dispatch-creation (linalg/linalg_ext stage)
    linalg_mlir = OUT_MLIR / f"{model_name}.mlir"
    log_file    = OUT_LOGS / f"{model_name}_exportLinalg.log"
    ok = _run(
        [
            "iree-compile",
            str(torch_mlir),
            "--iree-input-type=torch",
            "--compile-from=start",
            "--compile-to=input",
            # "--canonicalize", "--cse",
            "-o", str(linalg_mlir)
        ],
        log_file
    )
    if not ok:
        return  # 다음 모델로

    # 3) save inputs
    np.savez(
        OUT_INPUT / f"{model_name}.npz",
        **{f"arg{i}": t.cpu().numpy() for i, t in enumerate(tensors)}
    )
    print(f"✓ [{model_name}] linalg MLIR/params/input 완료")

def main() -> None:
    _mkdirs()
    ap = argparse.ArgumentParser(prog="export_to_linalg.py", description="IREE linalg MLIR export")
    ap.add_argument("--model", nargs="?", help="*_block.py stem in models/")
    ap.add_argument("--device", default="llvmcpu", help = "cuda | llvmcpu (default: llvmcpu)")
    args = ap.parse_args()

    targets = [args.model] if args.model_name else _discover_models()
    for tgt in targets:
        process(tgt, args.device)

if __name__ == "__main__":
    main()
