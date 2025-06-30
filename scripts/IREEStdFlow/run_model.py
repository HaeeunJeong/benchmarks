#!/usr/bin/env python3
import argparse, pathlib, numpy as np, safetensors.numpy as stnp
import iree.runtime as ireert

ROOT        = pathlib.Path(__file__).parent
OUT_PARAMS  = ROOT / "../../results/iree-output" / "params"
OUT_INPUT   = ROOT / "../../results/iree-output" / "input"
OUT_VMFB    = ROOT / "../../results/iree-output" / "vmfb"

def discover():
    return sorted(p.stem.replace(".safetensors","")
                  for p in OUT_PARAMS.glob("*_block.safetensors"))

def run_one(name: str, driver: str):
    reqs = [OUT_PARAMS/f"{name}.safetensors", OUT_VMFB/f"{name}.vmfb",
            OUT_INPUT/f"{name}.npz"]
    if not all(f.exists() for f in reqs):
        print(f"[{name}] 필요한 파일 없음")
        return

    cfg  = ireert.Config(driver); inst = cfg.vm_instance
    idx  = ireert.ParameterIndex()
    for k,v in stnp.load_file(reqs[0]).items():
        idx.add_buffer(k, v.tobytes())

    mods = ireert.load_vm_modules(
        ireert.create_io_parameters_module(inst, idx.create_provider("model")),
        ireert.create_hal_module(inst, cfg.device),
        ireert.VmModule.mmap(inst, str(reqs[1])),
        config=cfg)

    npz   = np.load(reqs[2]); args=[npz[k] for k in sorted(npz.files)]
    out   = mods[-1].main(*args)
    # print(f"▶ {name}: {out.to_host()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_name", nargs="?")
    ap.add_argument("--driver", default="local-sync")
    a = ap.parse_args()
    for n in ([a.model_name] if a.model_name else discover()):
        run_one(n, a.driver)

