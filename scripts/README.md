# Scripts

| Script | Platform | Purpose |
|--------|----------|---------|
| `install_torch_cuda.ps1` | Windows | Install CUDA-enabled PyTorch into `.venv` (also invoked from `run.bat` when NVIDIA GPU is detected) |
| `eval_accuracy.py` | Any | Leave-one-out accuracy evaluation over `data/refs/<tag>/`. Rebuilds the classifier per reference image (excluding that image from its own exemplars) and prints per-tag and overall accuracy. Needs the CLIP deps installed. |
| `verify_accuracy.py` | Any | Self-check of CLIP exemplar scoring, strict thresholds, and VLM merge mechanics. No GUI, no LM Studio, no reference photos required. |

Run the accuracy evaluation against the project venv:

```bash
.venv/Scripts/python scripts/eval_accuracy.py   # Windows
.venv/bin/python scripts/eval_accuracy.py        # Linux/macOS
```

Day-to-day use: **`run.bat`** (Windows) or **`run.sh`** (Linux/macOS), not files in this folder.
