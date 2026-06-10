# Pay-as-You-Decode: Selective High-Resolution Reconstruction from 1024× Compressed Latents

A hierarchical **VQ-VAE-2** for extreme image compression and generative reconstruction of
high-resolution grayscale images. A single image is encoded into a small stack of discrete
codebook indices; a learned autoregressive prior then lets you **decode detail on demand** —
store the heavily compressed top latents, and synthesize high-resolution structure only where
(and when) you need it.

<p align="center">
  <a href="https://www.youtube.com/watch?v=1rl-LlP0kVc">
    <img src="homepage.png" alt="Compression / reconstruction demo" width="900">
  </a>
  <br/>
  <a href="https://www.youtube.com/watch?v=1rl-LlP0kVc">
    <img src="https://img.shields.io/badge/YouTube-Watch-red?logo=youtube" alt="Watch on YouTube">
  </a>
</p>

---

## Highlights

- **Up to 1024× compression** of 1024×1024 images into a 32×32 grid of 8-bit discrete codes.
- **Hierarchical (two-level) quantization** (VQ-VAE-2 style) to trade compression for fidelity:
  a coarse *top* code map for global structure and a finer *bottom* map for local detail.
- **Learned transformer prior** over the latent grids — enables unconditional **generation**,
  latent in-painting, and *pay-as-you-decode*: keep only the 1024×-compressed top map and
  regenerate the bottom detail on demand instead of storing it.
- **Three training modes** behind one codebase, sharing the same encoder/decoder/quantizer.
- Pure **TensorFlow / Keras**, single- or multi-GPU, runs from 1024×1024 tiles.

## Method

```
image (1024×1024×1)
        │   encoder (strided conv stack + residual blocks)
        ▼
   continuous latents ──► vector quantizer ──► discrete code indices
        │                  (nearest codebook entry, straight-through est.)
        ▼
     decoder (transpose conv + residual blocks) ──► reconstruction
```

Quantization is the standard VQ objective: a reconstruction term plus a **codebook** and
**commitment** loss, with the straight-through estimator passing gradients across the
non-differentiable `argmin`. The two-level variant adds a second, higher-resolution code map
conditioned on the upsampled top latents, exactly as in VQ-VAE-2. A separate **decoder-only
transformer** is then trained as an autoregressive prior over the code grids, turning the model
into a generator and enabling selective decoding.

### Compression modes

| Mode | Config | Latents (per 1024² image) | Compression | What it's for |
|------|--------|---------------------------|-------------|----------------|
| **1 — Top-only** | `top_only` | 32×32 × 8-bit | **1024×** | Maximum compression; global structure |
| **2 — Two-level** | `two_level` | 32×32 + 64×64 × 8-bit | **~204×** | High-fidelity reconstruction |
| **3 — Two-level + prior** | `two_level_with_prior` | top stored, bottom generated | **1024× stored** | Generation + pay-as-you-decode |

> Compression is exact arithmetic on the discrete codes: a 1024×1024×8-bit image is 8.39 Mbit;
> the top map is 32×32×log₂256 = 8.19 Kbit → **1024×**. Adding the 64×64 bottom map gives
> 40.96 Kbit → **~204×**. Mode 3 keeps only the top map on disk and reconstructs the bottom
> detail from the prior, recovering Mode-2 fidelity at Mode-1 storage cost.

## Installation

```bash
pip install -r vqvae_requirements.txt
```

Verify the install:

```bash
python3 -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
python3 -c "from vqvae_modules import VQVAETopOnly; print('imports OK')"
python3 test_vqvae_setup.py        # builds a model and runs a forward pass
```

A virtual environment is recommended:

```bash
python3 -m venv vqvae_env && source vqvae_env/bin/activate
pip install -r vqvae_requirements.txt
```

For GPU acceleration install the matching CUDA build (`pip install tensorflow[and-cuda]`),
or on Apple Silicon `pip install tensorflow-macos tensorflow-metal`.

## Usage

Each mode has a self-contained training script. Point `--data_dir` at a folder of images
(large images are tiled into fixed-size patches automatically).

```bash
# Mode 1 — top-only, 1024× compression
python3 train_mode1_toponly.py --data_dir /path/to/images --output_dir ./runs/mode1 --epochs 100

# Mode 2 — two-level, ~204× compression
python3 train_mode2_twolevel.py --data_dir /path/to/images --output_dir ./runs/mode2 --epochs 150

# Mode 3 — transformer prior over Mode-2 latents (needs a trained Mode-2 checkpoint)
python3 train_mode3_prior.py --data_dir /path/to/images --output_dir ./runs/mode3 \
        --pretrained_vqvae ./runs/mode2/best_model

# Evaluate a trained model (reconstruction metrics + samples)
python3 evaluate_vqvae.py --model_dir ./runs/mode2 --data_dir /path/to/images
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir ./runs/mode1/tensorboard
```

Hyperparameters (image size, codebook sizes, grid resolutions, learning rate, …) live in
`vqvae_modules/config.py` as `get_config('top_only' | 'two_level' | 'two_level_with_prior')`.

### Out of memory?

Lower the batch size (`--batch_size 1`) or train at a smaller `image_size`. For 1024×1024 on
8 GB VRAM, batch size 2 is a good starting point; enable mixed precision for a speedup:

```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

## Repository layout

```
vqvae_modules/
  models.py        VQ-VAE encoder / decoder / quantizer + two-level model
  layers.py        residual blocks, vector-quantizer layer, building blocks
  data_utils.py    image loading, tiling, tf.data pipelines
  eval_utils.py    reconstruction metrics and sample helpers
  config.py        per-mode hyperparameter presets
train_mode1_toponly.py     Mode 1 trainer (1024×)
train_mode2_twolevel.py    Mode 2 trainer (~204×)
train_mode3_prior.py       Mode 3 trainer (transformer prior)
evaluate_vqvae.py          evaluation / sampling
vqvae2_compression.ipynb   walkthrough of the three modes
```

## Requirements

- Python 3.8+ (3.10+ recommended)
- TensorFlow 2.10+
- A GPU with ≥8 GB VRAM is recommended for 1024×1024 training (CPU works but is much slower)

## Contact

Questions, issues, or ideas are welcome — open an issue or PR, or reach out at
📧 fumingyang@fas.harvard.edu
