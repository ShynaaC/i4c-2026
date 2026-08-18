# KLA AI Image Restoration — i4C Hackathon 2026

Submission for **Track 1: AI-Based Restoration of Degraded Images for Semiconductor Inspection**. This repository contains a complete, directly runnable pipeline for removing combined speckle/Gaussian degradation and restoring grayscale images to exactly 2× their input resolution.

## Reviewer quick start

The bundled checkpoint is only 491 KB and is tracked directly in this repository. No download, path edit, notebook, or training step is needed for inference.

### 1. Clone and install

Python 3.12 is recommended. PyTorch's CUDA wheel is selected through `requirements.txt` on a CUDA-capable Linux host.

```bash
git clone https://github.com/ShynaaC/i4c-2026.git
cd i4c-2026

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows activation equivalent:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Run inference

```bash
python evaluate.py /absolute/path/to/test/NoisyLR /absolute/path/to/output
```

That is the complete evaluation command. `evaluate.py` automatically:

- loads `weights/restoration_model.pt` relative to the script, independent of the current working directory;
- selects CUDA when available and otherwise runs on CPU;
- enables FP16 on CUDA for faster H100 inference;
- discovers every supported input recursively and processes same-sized images in batches;
- cleans NaN/Inf values, clips the noisy input to the trained `[0, 1]` range, and restores exactly 2× spatial resolution;
- writes each result under the same relative filename in the requested output directory.

For the challenge-provided `.npy` files, outputs are 2D `float32` arrays in `[0, 1]`. A `128×128` input becomes `256×256`; a `256×256` input becomes `512×512`. `.png`, `.tif`, `.tiff`, `.bmp`, `.jpg`, and `.jpeg` grayscale inputs are also accepted, with their original extensions preserved.

Useful optional flags:

```bash
python evaluate.py INPUT_DIR OUTPUT_DIR --batch-size 16
python evaluate.py INPUT_DIR OUTPUT_DIR --device cuda
python evaluate.py INPUT_DIR OUTPUT_DIR --weights /path/to/checkpoint.pt
```

Use a smaller batch size if GPU memory is constrained. Run `python evaluate.py --help` for the complete interface.

## Repository contents

| Path | Purpose |
|---|---|
| `evaluate.py` | Standalone evaluation script used as-is with input and output directory arguments |
| `train.py` | Reproducible training script for paired `.npy` data |
| `model.py` | Shared compact restoration architecture |
| `weights/restoration_model.pt` | Final trained PyTorch checkpoint |
| `restored_outputs/` | All 400 restored challenge test arrays |
| `requirements.txt` | Pinned Python training/inference environment |

Checkpoint SHA-256: `CAF11C510350C1369B6E36ABE5FF79FF1DD5DAFA4893006DA4290964C696EB56`

## Model

`FastRestoreNet` is a compact residual super-resolution network designed for speed and OOD stability:

1. The input is made finite and clipped to the physically valid `[0, 1]` range, preventing speckle outliers from destabilizing inference.
2. Four residual blocks at low resolution learn joint denoising and structure extraction.
3. A pixel-shuffle head performs learned 2× upsampling efficiently.
4. A bicubic skip path supplies a conservative image estimate; the network predicts only a bounded correction. This reduces hallucination risk on unseen semiconductor structures.

The model has approximately 121,000 parameters and a 491 KB checkpoint. It accepts one grayscale channel and returns one grayscale channel.

Training uses paired random crops, horizontal/vertical flips, transpose augmentation, AdamW, a cosine learning-rate schedule, gradient clipping, and a Charbonnier reconstruction loss plus an edge-gradient loss. The edge term encourages sharp structures without relying on adversarial training, which can invent inspection details.

## Reproduce training from scratch

Extract the supplied training archive so that the directories look like this:

```text
/path/to/train/
├── GT/
│   ├── 000000.npy
│   └── ...
└── NoisyLR/
    ├── 000000.npy
    └── ...
```

The stem of every low-resolution input must match its ground truth. To reproduce the submitted one-epoch checkpoint and deterministic split:

```bash
python train.py \
  --data-root /path/to/train \
  --output weights/restoration_model.pt \
  --epochs 1 \
  --batch-size 16 \
  --crop-size 96 \
  --val-count 16 \
  --seed 2026
```

For a longer run, increase `--epochs` (the script default is 8). Training automatically uses CUDA and mixed precision when available. It validates after every epoch and saves only the best checkpoint. Pairing, dimensions, and missing files are checked before training begins.

## Validation and speed checks

The submitted checkpoint was trained from the 3,200 provided pairs, holding out 16 pairs selected with seed 2026. On that fixed smoke split:

| Method | PSNR |
|---|---:|
| Clipped bicubic baseline | 22.147 dB |
| Submitted FastRestoreNet | 25.362 dB |
| Improvement | **+3.215 dB** |

There is no test-set ground truth in this repository, so no unsupported test quality claim is made. All 400 supplied test inputs were processed successfully. Local CPU inference took 9.595 seconds total (0.0240 seconds/image); the official H100 timing will differ and is expected to be substantially faster.

The committed outputs were programmatically checked for exact filename coverage, 2× dimensions, `float32` dtype, finite values, and `[0, 1]` range: 400 inputs, 400 outputs, zero missing, zero extra, zero invalid.

## Environment

The checkpoint was produced with Python 3.12.13, NumPy 2.3.5, Pillow 12.3.0, and PyTorch 2.7.1. The requirements are fully pinned, including PyTorch's runtime dependencies. The checkpoint contains plain tensors and metadata only and is loaded with PyTorch's safe `weights_only=True` mode.
