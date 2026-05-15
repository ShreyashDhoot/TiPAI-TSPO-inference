# TiPAI-TSPO: Adversarial Auditing & Latent Reinsertion

TiPAI-TSPO (Tournament-based Safety Policy Optimization) is a high-performance inference pipeline designed to audit and "steer" text-to-image generation toward safety. It detects adversarial violations (nudity, violence, etc.) mid-generation and uses a tournament-based selection process to inpaint and reinsert safe content without destroying the image trajectory.

## Features
- **Multi-Family Support**: Native support for **SD 1.5**, **SDXL**, and **FLUX.1** (Flow-matching).
- **Adversarial Audit Tournament**: Mid-generation pausing to evaluate candidates based on Safety (Auditor), Intent (Policy), and Visual Quality (Faithfulness).
- **Flow-Compatible Reinsertion**: A custom latent-space bridge specifically designed for Flow-Matching models (FLUX/SD3.5) to prevent trajectory collapse during edits.
- **Single-GPU Optimized**: Runs massive 33GB models (like FLUX.1) on a single 48GB VRAM setup using:
  - Model CPU Offloading
  - VAE Tiling & Slicing
  - bfloat16 Precision

## Technical Innovation: Flow-Compatible Reinsertion
Standard DDIM/DDPM reinsertion often fails on Flow-Matching architectures (FLUX.1 / SD 3.5), resulting in incomprehensible noise. This pipeline implements a **Linear Flow Interpolation Bridge**:
1. **Unpacking**: Converts packed FLUX latents into pixel space for auditing.
2. **Editing**: Performs safety-steered inpainting using a specialized auditor tournament.
3. **Repacking & Blending**: Scales and repacks edited pixels back into the structural grid of the Flow-transformer, using a linear velocity approximation to maintain trajectory consistency.

## Setup

1. **Environment**:
   ```bash
   # Create and activate venv
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Authentication**:
   Create a `.env` file in the root directory:
   ```env
   HF_TOKEN=your_huggingface_token_here
   ```

3. **Configuration**:
   Edit `tipai/config.yaml` to select your base model and audit schedule.

## Running Inference

```bash
python3 tipai/run.py --prompt "a cinematic photo of a neon sign" --config tipai/config.yaml
```

## Results
Outputs are saved in the `results/` directory:
- `*_safe.png`: The final audited/intervened image.
- `tournament_results/`: Visual diagnostic grids showing the candidate selection process.
