#!/usr/bin/env python3
import argparse, importlib, pathlib, numpy as np, torch
from safetensors.torch import save_file
import iree.turbine.aot as aot

import sys
# print(sys.path)
sys.path.append(str(pathlib.Path('models').resolve().parent))
# print(sys.path)

ROOT        = pathlib.Path(__file__).parent
# print(f"ROOT: {ROOT}")
MODELS_DIR  = ROOT / "../../models"
OUT_PARAMS  = ROOT / "../../results/iree-output" / "params"
OUT_MLIR    = ROOT / "../../results/iree-output" / "mlir"
OUT_INPUT   = ROOT / "../../results/iree-output" / "input"

def mkdirs():  # ensure dirs
    for d in (OUT_PARAMS, OUT_MLIR, OUT_INPUT):
        d.mkdir(parents=True, exist_ok=True)

def discover_models():
    return sorted(p.stem for p in MODELS_DIR.glob("*_block.py"))

def _tensorize(d):  # tuple→rand tensor
    if isinstance(d, tuple) and all(isinstance(x, int) for x in d):
        return torch.randn(d)
    return d
    # return torch.randn(d) if isinstance(d, tuple) else d
def _flatten(x):
    return sum((_flatten(e) if isinstance(x, (tuple, list)) 
        else [x] for e in (x if isinstance(x,(tuple,list)) else [x])), [])

def process(mname: str):
    mod   = importlib.import_module(f"models.{mname}")
    model = mod.get_model().eval()

    # 1) params
    save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()},
              OUT_PARAMS / f"{mname}.safetensors")

    # 2) externalize + MLIR
    aot.externalize_module_parameters(model)
    dummy   = _tensorize(mod.get_dummy_input())
    tensors = _flatten(dummy)
    aot.export(model,*tensors).save_mlir(OUT_MLIR / f"{mname}.mlir")





    # 3) inputs
    np.savez(OUT_INPUT / f"{mname}.npz",
             **{f"arg{i}": t.cpu().numpy() for i,t in enumerate(tensors)})
    print(f"✓ [{mname}] params/mlir/input 저장")

if __name__ == "__main__":
    mkdirs()
    ap = argparse.ArgumentParser()
    ap.add_argument("model_name", nargs="?", help="models/ 의 *_block.py stem")
    arg = ap.parse_args()
    targets = [arg.model_name] if arg.model_name else discover_models()
    for t in targets: process(t)

