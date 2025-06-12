# Benchmarks: PyTorch → StableHLO → IREE

A **reproducible micro‑benchmark suite** for testing how common PyTorch models compile and run across multiple back‑ends (CPU, CUDA, Vulkan, ROCm) using the OpenXLA / IREE toolchain.

---

## 📁 Repository layout

```text
benchmarks/
├── env/                 # Pre‑baked Conda env specs (pick one)
│   ├── torch-stablehlo.yaml        # PyTorch + StableHLO (CPU)
│   ├── torch-stablehlo-gpu.yaml    # PyTorch + StableHLO (CUDA)
│   └── stablehlo-iree.yaml         # Stand‑alone StableHLO → IREE toolchain
├── models/              # Lightweight model stubs (Conv, GAT …)
│   ├── conv_block.py
│   ├── gcn_block.py
│   ├── graphsage_block.py
│   ├── gat_block.py
│   └── gatv2_block.py
├── scripts/             # All entry‑point helpers
│   ├── run_bench.py     # Pure‑PyTorch latency probe
│   └── compile_iree.py  # PyTorch → StableHLO → IREE runner /// Not Working Currently
└── results/             # *.vmfb, *.csv outputs are dropped here
```

---

## 🚀 Quick start

### 1 · Clone & create env (Python 3.11)

```bash
# clone your private repo
$ git clone git@github.com:HaeeunJeong/benchmarks-for-my-compiler.git && cd benchmarks-for-my-compiler

# conda (choose one of the YAML files)
# CPU‑only example:
$ conda env create -f env/torch-stablehlo.yaml -n pt-bench
# CUDA  GPU example:
# $ conda env create -f env/torch-stablehlo-gpu.yaml -n pt-bench
$ conda activate pt-bench
```

<details>
<summary>…or Docker</summary>

```bash
$ docker build -t pt-bench -f env/Dockerfile .
$ docker run --gpus all -it pt-bench /bin/bash
```

</details>

### 2 · Run the plain PyTorch micro‑benchmarks

```bash
# run all models on CPU
$ python -m scripts.run_bench --device cpu --csv

# a single model on GPU
$ python -m scripts.run_bench resnet --device cuda
```

*Outputs*: per‑model latency is printed and appended to `results/latency.csv` (timestamp, model, device, time / "error").

---

## 🏗️ Supported models

| Category        | Key                 | Source                                                | Notes                                                                         |
| --------------- | ------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------- |
| Simple custom   | `conv`              | `models/conv_block.py`                                | 2×Conv + ReLU toy block used as a sanity‑check kernel                         |
| **GNN**         | `gcn`\*             | `models/gcn_block.py`                                 | Graph Convolution Net                                                         |
| **GNN**         | `graphsage`\*       | PyTorch Geometric                                     | Graph Sage Net; The milestone of GNN model                                    |
| **GNN**         | `gat`\*             | PyTorch Geometric                                     | Graph Attention Net; contains `scatter_reduce`/`nonzero` (export‑unsupported) |
| **GNN**         | `gatv2`\*           | PyTorch Geometric                                     | Graph Attention Net; contains `scatter_reduce`/`nonzero` (export‑unsupported) |
| **CNN**         | `resnet`            | `torchvision` ResNet‑18                               | Classic image backbone; ImageNet pretrained                                   |
| **CNN**         | `mobilenet`         | `torchvision` MobileNet v3 S                          | Mobile‑oriented CNN; ImageNet pretrained                                      |
| **Transformer** | `vit`               | `torchvision` ViT‑B/16                                | Vision Transformer baseline                                                   |
| **Transformer** | `bert`\*            | `bert-base-uncased`                                   | Token‑level encoder; needs full kwargs to export                              |
| **Transformer** | `gpt2`\*            | `gpt2-xl`                                             | 1.5 B‑param decoder; export kwargs WIP                                        |
| **LLM**         | `llama`\*           | `meta-llama/Llama-3.2-3B`                             | Base Llama 3.2 3 B; compact general‑purpose LLM                               |
| **LLM**         | `llama-qlora`\*     | `meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8`     | 4‑bit QLoRA; memory‑efficient inference                                       |
| **LLM**         | `llama-spinquant`\* | `meta-llama/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8` | 4‑bit SpinQuant; alt quantization scheme                                      |
| **LLM**         | `deepseek`\*        | `DeepSeek‑R1‑Distill‑Qwen`                            | Distilled math‑centric LLM; complex TF graph                                  |

<sup>\* asterisked entries are currently skipped in `compile_iree.py` (see `UNSUPPORTED`).</sup>
<sup>\* marked entries are currently skipped in compile\_iree.py (see `UNSUPPORTED`).</sup>

---

## 🔧 Compiling & running with IREE

```bash
# compile all supported vision models to LLVM‑CPU & store vmfb
$ python -m scripts.compile_iree --target llvm-cpu --csv

# pick a specific model & CUDA back‑end (needs GPU + CUDA driver)
$ python -m scripts.compile_iree resnet mobilenet --target cuda
```

*Outputs*

* **FlatBuffer (`.vmfb`)**: saved to `results/<model>-<target>.vmfb`
* **Latency CSV**: appended to `results/iree_latency.csv` with status `time (ms)` / `error` / `unsupported`.

---

## ➕ Adding new models

1. Drop a new Python module under `models/`, e.g. `my_model.py`.
2. Expose `(model, dummy_input)` via a helper (see existing examples).
3. Add its key to `scripts/run_bench.py` → `load_model()` and to `ALL_MODELS`.
4. If the model exports cleanly to StableHLO, remove it from `UNSUPPORTED` in `compile_iree.py`.

---

## 🛠️ Troubleshooting

| Symptom                                           | Fix                                                                                                 |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `UserWarning: 'pretrained' is deprecated`         | TorchVision ≥0.13 uses `weights=..._Weights.DEFAULT`; already handled.                              |
| `This model contains ops not capturable ...`      | Operation not yet supported by Torch‑XLA; keep model in `UNSUPPORTED` or write a Torch‑FX fallback. |
| `treespec.unflatten` during export                | Provide full kwargs (`attention_mask`, etc.) matching the model’s forward signature.                |
| `expected operation name in quotes` at IREE stage | Ensure you pass **the actual MLIR string**, not a Python repr (use `str(shlo.mlir_module)`).        |

---

## 📜 License / visibility

This is a **private** research benchmark. Do **NOT** publish benchmark numbers externally without permission.

