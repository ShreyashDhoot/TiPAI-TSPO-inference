# TiPAI-TSPO — Modular Safe Diffusion

End-to-end pipeline: send a prompt → receive a safe image.

```
tipai/
├── config.yaml                    ← all knobs in one place
├── run.py                         ← CLI entry-point
│
├── auditor/auditor.py             ← adversarial auditor inference
├── policy/tspo_policy.py          ← TSPO policy (knob proposals)
├── inpainting/inpainter.py        ← SD-inpaint wrapper (swappable LoRA)
├── reinsertion/reinsertion.py     ← 4 reinsertion methods (SD0–SD3)
├── tournament/winner.py           ← guarded utility + winner selection
├── pipeline/safe_diffusion.py     ← main loop (orchestrates everything)
└── utils/
    ├── config_loader.py
    └── diffusion_utils.py
```

---

## Quick start

```bash
# install deps (once)
pip install diffusers transformers accelerate torch torchvision \
            opencv-python-headless lpips peft pyyaml

# download auditor weights (HuggingFace)
wget https://huggingface.co/kricko/Adversarial-Image-Auditor-v2/resolve/main/complete_auditor_best.pth
wget https://huggingface.co/kricko/Adversarial-Image-Auditor-v2/resolve/main/vocab.json

# generate a safe image
python run.py --prompt "a woman at the beach"
```

---

## Config (config.yaml)

| Key | Purpose |
|-----|---------|
| `base_sd_model` | HF id or local path for the base SD1.x model |
| `inpainter_model` | HF id or local path for the inpainter |
| `inpainter_lora_path` | Path to inpainter LoRA weights (null = skip) |
| `auditor_weights` | Path to `complete_auditor_best.pth` |
| `auditor_vocab` | Path to `vocab.json` |
| `use_tspo` | `true` → TSPO policy, `false` → vanilla sweeps |
| `tspo_checkpoint` | Path to trained TSPO `.pth` |
| `reinsertion_method` | `SD0_DDPM` / `SD1_DDIM_FWD` / `SD2_DDIM_INV` / `SD3_NULL_TEXT` |
| `audit_steps` | List of denoising step indices to audit |
| `n_candidates` | Candidates per tournament |
| `delta` / `tau_P` / `tau_F` | Guarded utility thresholds |

---

## Swapping components at runtime

```python
from utils.config_loader import load_config
from pipeline.safe_diffusion import SafeDiffusionPipeline

cfg = load_config("config.yaml")
sdp = SafeDiffusionPipeline(cfg)

# swap base SD model (keeps shared VAE)
sdp.swap_base_model("stabilityai/stable-diffusion-v1-5")

# swap inpainter LoRA (no pipeline reload needed)
sdp.swap_inpainter_lora("weights/my_custom_lora.safetensors", lora_scale=0.9)

result = sdp.generate("a family having a picnic")
result.image.save("output.png")
```

---

## Module responsibilities

### `auditor/auditor.py`
- `AdversarialAuditor.audit_pil(pil, prompt, t_norm)` → `AuditResult`
- Returns: `adv_prob`, `is_unsafe`, `harm_class`, `policy_score`, `faithfulness`, `seam_quality`, `mask_pil`, `heatmap`

### `policy/tspo_policy.py`
- `load_policy(ckpt_path, device)` → `TSPOPolicy | None`
- `get_knobs(policy, audit_result, t_norm, harm_class, n)` → `list[KnobSet]`
- Falls back to hardcoded vanilla sweeps when `policy=None`

### `inpainting/inpainter.py`
- `build_inpainter(model_id, lora_path, ...)` → `StableDiffusionInpaintPipeline`
- `run_inpainting(pipe, base_pil, mask_pil, prompt, harm_class, knob)` → `PIL.Image`
- `swap_lora(pipe, new_lora_path)` — hot-swap LoRA without reloading

### `reinsertion/reinsertion.py`
- `reinsert(method, pipe, base_latents, winner_pil, mask_pil, ...)` → `torch.Tensor`
- Only `SD3_NULL_TEXT` is used in production (spec requirement)
- Others (SD0–SD2) retained for ablations

### `tournament/winner.py`
- `guarded_utility(candidate_res, control_res, cfg)` → `float`
- `select_winner(candidates, scores, control_pil, control_res, cfg)` → winner + metadata
- Returns control image if all candidates are rejected

### `pipeline/safe_diffusion.py`
- `SafeDiffusionPipeline(cfg).generate(prompt)` → `GenerationResult`
- `GenerationResult.image` — final safe PIL image
- `GenerationResult.metrics` — safety metrics dict
- `GenerationResult.trajectory` — per-step intervention log
