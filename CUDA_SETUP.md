# TensorFlow + PyTorch CUDA Co-installation Guide

A practical guide for running TensorFlow and PyTorch in the same Python virtual environment with GPU (CUDA) support.

---

## Compatibility Matrix

The single most important constraint is that **both frameworks must target the same CUDA major.minor version**. Use the table below to choose a stable combination.

| CUDA | cuDNN | TensorFlow | PyTorch | Python |
|------|-------|-----------|---------|--------|
| 12.1 | 8.9   | 2.13 – 2.16 | 2.1 – 2.3 | 3.9 – 3.11 |
| 11.8 | 8.7   | 2.10 – 2.13 | 2.0 – 2.1 | 3.8 – 3.11 |

> **Recommended stable combination (2024/2025):**  
> CUDA 12.1 · cuDNN 8.9 · TensorFlow 2.15 · PyTorch 2.2 · Python 3.10

---

## Prerequisites

- NVIDIA driver ≥ 525 (for CUDA 12.x) — check with `nvidia-smi`
- Python 3.10 (recommended)
- `venv` or `conda`

---

## Installation — Standard (public PyPI + PyTorch index)

```bash
# 1. Create and activate the virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install TensorFlow (includes CUDA/cuDNN wheels since TF 2.12)
pip install "tensorflow==2.15.*"

# 4. Install PyTorch with matching CUDA 12.1 wheels
pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu121
```

---

## Installation — Enterprise Repository

If your network requires routing through an enterprise PyPI mirror (e.g. `https://repo-python.prod.adp.fr/root/pypi`), apply one of the two options below.

### Option A — Enterprise mirror only

Use this when the mirror already hosts the correct GPU wheels for both packages.

```bash
sudo -u app -H bash -lc '
cd /srv/applications/app-n1960-api/bin &&
source .venv/bin/activate &&
mkdir -p .tmp .cache &&
TMPDIR=$(pwd)/.tmp \
pip install \
  --cache-dir $(pwd)/.cache \
  --index-url https://repo-python.prod.adp.fr/root/pypi \
  "tensorflow==2.15.*" torch torchvision
'
```

### Option B — Enterprise mirror + PyTorch CUDA fallback (recommended)

Use this when the mirror does not host PyTorch GPU wheels. `pip` will prefer the enterprise mirror for all packages, and fall back to the official PyTorch CUDA index only for torch wheels that are not found there.

```bash
sudo -u app -H bash -lc '
cd /srv/applications/app-n1960-api/bin &&
source .venv/bin/activate &&
mkdir -p .tmp .cache &&
TMPDIR=$(pwd)/.tmp \
pip install \
  --cache-dir $(pwd)/.cache \
  --index-url  https://repo-python.prod.adp.fr/root/pypi \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  "tensorflow==2.15.*" torch torchvision
'
```

> **Note:** If network policy blocks `download.pytorch.org`, the enterprise mirror must explicitly host PyTorch CUDA wheels (e.g. `torch-2.2.0+cu121-cp310-cp310-linux_x86_64.whl`). Without the correct GPU wheel, pip will silently install a CPU-only build.

---

## Uninstalling a Bad PyTorch Installation

If a previous install left inconsistent NVIDIA sub-packages, clean them before reinstalling:

```bash
pip uninstall -y \
  torch torchvision torchaudio triton \
  nvidia-nccl-cu12 nvidia-nvtx-cu12 nvidia-cuda-runtime-cu12 \
  nvidia-cuda-nvrtc-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 \
  nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-cuda-cupti-cu12 \
  nvidia-nvjitlink-cu12
```

---

## Verification

### TensorFlow

```bash
python - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs found:", len(gpus), gpus)
PY
```

Expected output (GPU build): `GPUs found: 1 [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

### PyTorch

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    x = torch.rand(1024, 1024, device="cuda")
    y = torch.rand(1024, 1024, device="cuda")
    z = x @ y
    print("GPU matmul OK:", z.shape, z.device)
PY
```

Expected output: `CUDA available: True`, `Device count: 1`, `GPU matmul OK: torch.Size([1024, 1024]) cuda:0`

### Both together

```bash
python - <<'PY'
import tensorflow as tf
import torch
print("TensorFlow:", tf.__version__, "| GPUs:", len(tf.config.list_physical_devices('GPU')))
print("PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda, "| available:", torch.cuda.is_available())
PY
```

---

## Common Issues and Fixes

### `ImportError` / `OSError` on `torch` import after install

**Symptom:** `torch` was installed successfully but import raises an error mentioning a missing `.so` or `libcuda`.

**Causes and fixes:**

| Cause | Fix |
|---|---|
| CPU-only wheel installed (no `+cu121` suffix in wheel name) | Reinstall with `--index-url https://download.pytorch.org/whl/cu121` or ensure the enterprise mirror hosts the GPU wheel |
| NVIDIA driver too old for CUDA 12.x | Use CUDA 11.8 wheels (`cu118`) and `tensorflow==2.13.*` |
| `triton` version mismatch | `pip install --upgrade triton` or pin to `triton==2.2.0` |
| Missing system CUDA toolkit | Not required for PyTorch ≥ 2.0 (ships its own CUDA libs), but required for TF < 2.12 |

### NCCL error on multi-GPU

```
RuntimeError: NCCL error: unhandled system error
```

The `nvidia-nccl-cu12` wheel shipped with PyTorch may conflict with a system NCCL library. Fix: uninstall the system NCCL (`apt remove libnccl*`) and let PyTorch use its bundled version, **or** pin the PyTorch NCCL wheel to match the system version.

### TensorFlow and PyTorch loading different cuDNN versions

Both frameworks now ship their own cuDNN wheels and load them at import time, so conflicts are rare with modern versions (TF ≥ 2.12, PyTorch ≥ 2.0). If you see cuDNN-related `OSError`, run:

```bash
python -c "import ctypes; ctypes.CDLL('libcudnn.so.8')"
```

and compare the resolved path with `find .venv -name 'libcudnn*'`. If multiple versions coexist, pin both frameworks to the same cuDNN major version.

---

## Key Constraints Summary

1. **CUDA version must match** between both frameworks — choose CUDA 12.1 (for latest) or CUDA 11.8 (for broader driver compatibility).
2. **Python 3.10** is the safest choice for both TF 2.15 and PyTorch 2.2.
3. **TF ≥ 2.12 and PyTorch ≥ 2.0** both bundle their own CUDA/cuDNN shared libraries, eliminating the need for a system-level CUDA toolkit installation.
4. When using an enterprise mirror, **verify that GPU wheels are present** (`+cu121` or `+cu118` build tag in the wheel filename). A missing GPU wheel results in a silent CPU-only install.
5. Installing via `--extra-index-url` means pip may pick a wheel from either index — always check `pip show torch` after install to confirm the installed version includes the CUDA tag.

---

## References

- [TensorFlow install guide](https://www.tensorflow.org/install/pip)
- [PyTorch get started](https://pytorch.org/get-started/locally/)
- [PyTorch CUDA wheel index](https://download.pytorch.org/whl/cu121)
- [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
