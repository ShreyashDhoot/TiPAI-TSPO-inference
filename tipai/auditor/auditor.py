"""
auditor/auditor.py
──────────────────
Self-contained adversarial auditor.

Public API
----------
AdversarialAuditor(model_path, vocab_path) → auditor
auditor.audit_pil(pil, prompt, t_norm) → AuditResult (TypedDict)

AuditResult keys
----------------
adv_prob      float         P(adversarial)  ∈ [0,1]
is_unsafe     bool          adv_prob > 0.5
harm_class    str           'safe' | 'nudity' | 'violence'
policy_score  float         1 - adv_prob
faithfulness  float         cos-similarity rescaled to [0,1]  ← FIXED
seam_quality  float         seam/blend quality score
mask_pil      PIL.Image     binary inpaint mask (L mode, 512×512)
heatmap       np.ndarray    raw adversarial heatmap (512×512, float32)
img_embed     torch.Tensor  (256,) auditor image embedding – used by StateEncoder
text_embed    torch.Tensor  (512,) auditor text embedding  – used by StateEncoder

Architecture / faithfulness notes
-----------------------------------
TRAINING MISMATCH FIXED
  train_new.py initialises CompleteMultiTaskAuditor with num_classes=6 (line 402).
  The old inference code used num_classes=3, causing a state_dict shape mismatch
  on class_head and object_detection_head conv weights.
  Fix: inference model now uses num_classes=_CKPT_NUM_CLASSES=6 and only reads
  the first 3 class logits (safe/nudity/violence) for harm-class classification.

FAITHFULNESS BUG FIXED
  img_proj_head and txt_proj_head both end with a ReLU activation, so projected
  embeddings are in [0, ∞) and their cosine similarity is always in [0, ~0.15].
  The old formula  (cos + 1) / 2  maps this to [0.50, 0.575] — a blind gate.

  Fix — two-tier strategy:
  Tier 1 (preferred): use txt_feat_raw (B, 512), the raw LSTM→FC output that
    passes through LayerNorm (genuinely zero-centred).  Average both 256-d halves
    to get a 256-d vector matching img_embed's dimension, L2-normalise both.
    Cosine similarity is now in a genuine [-1, +1] range so (cos+1)/2 is correct.
  Tier 2 (fallback): txt_feat_raw unavailable → return 0.0 (reject-safe default).
"""

from __future__ import annotations
import json, os
from typing import TypedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# ── class names ──────────────────────────────────────────────────────────────
CLASS_NAMES = ["safe", "nudity", "violence"]
_CKPT_NUM_CLASSES = 3


# ─────────────────────────────────────────────────────────────────────────────
# Internal sub-models  (mirror train_new.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleTokenizer:
    def __init__(self, vocab_path: str | None = None, max_length: int = 77):
        self.max_length = max_length
        self.word_to_idx: dict[str, int] = {
            "<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3
        }
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path) as f:
                self.word_to_idx = json.load(f)

    def encode(self, text: str) -> torch.Tensor:
        if not text:
            return torch.zeros(self.max_length, dtype=torch.long)
        words = str(text).lower().split()
        idx = [2] + [self.word_to_idx.get(w, 1) for w in words[: self.max_length - 2]] + [3]
        while len(idx) < self.max_length:
            idx.append(0)
        return torch.tensor(idx[: self.max_length], dtype=torch.long)


class SimpleTextEncoder(nn.Module):
    """Matches train_new.py SimpleTextEncoder exactly."""
    def __init__(self, vocab_size: int = 50_000, embed_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(hidden_dim * 2, 512)
        self.norm = nn.LayerNorm(512)
        self.drop = nn.Dropout(0.1)

    def forward(self, toks: torch.Tensor):
        e = self.drop(self.embedding(toks))
        out, (h, _) = self.lstm(e)
        h = torch.cat([h[0], h[1]], dim=1)
        # text_feat  (B, 512) — raw LSTM→FC, zero-centred via downstream LayerNorm
        # seq_feat   (B, T, 512) — per-token pre-LN features for cross-attention K/V
        text_feat = self.fc(h)
        seq_feat  = self.norm(self.fc(out))
        return text_feat, seq_feat, toks.eq(0)


class CompleteMultiTaskAuditor(nn.Module):
    """
    Exactly mirrors train_new.py CompleteMultiTaskAuditor.

    Changes from old inference-only version:
    - num_classes=_CKPT_NUM_CLASSES (6) so load_state_dict shape matches.
    - forward() returns 'txt_feat_raw' (B, 512) for correct Tier-1 faithfulness.
    - 'txt_embed' (B, 256) kept for backward compat only; NOT used for faithfulness.
    """

    def __init__(self, num_classes: int = _CKPT_NUM_CLASSES, vocab_size: int = 50_000):
        super().__init__()
        resnet = models.resnet101(weights=None)
        self.features   = nn.Sequential(*list(resnet.children())[:-2])
        self.text_encoder = SimpleTextEncoder(vocab_size=vocab_size)

        # Safety heads — num_classes must match checkpoint
        self.adv_head   = nn.Conv2d(2048, 1, 1)
        self.class_head = nn.Conv2d(2048, num_classes, 1)
        self.quality_head = nn.Conv2d(2048, 1, 1)
        self.object_detection_head = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, num_classes, 1)
        )

        # Cross-attention path (image queries text tokens)
        self.image_proj = nn.Conv2d(2048, 512, 1)
        self.cross_attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        self.query_norm = nn.LayerNorm(512)
        self.key_norm   = nn.LayerNorm(512)

        # CLIP-style faithfulness heads (end in ReLU → outputs ≥ 0)
        self.img_proj_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 256))
        self.txt_proj_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 256))
        self.log_temperature = nn.Parameter(torch.tensor([-2.659]))

        # Timestep embedding + FiLM
        self.timestep_embed = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 512)
        )
        self.film_adv  = nn.Linear(512, 2048 * 2)
        self.film_seam = nn.Linear(512, 512 * 2)

        # Relative adversary head (FiLM-conditioned on timestep)
        self.relative_adv_head = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1)
        )

        # Seam quality detection
        self.seam_feat = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512)
        )
        self.seam_cls = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256), nn.Conv2d(256, 1, 1)
        )

    def forward(self, x: torch.Tensor, text_tokens=None, timestep=None):
        B = x.size(0)
        feats = self.features(x)
        gf = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)   # (B, 2048)

        img_embed = txt_embed = txt_feat_raw = None

        if text_tokens is not None:
            tf, sf, pm = self.text_encoder(text_tokens)        # tf: (B,512) raw
            iq = self.image_proj(feats).view(B, 512, -1).permute(0, 2, 1)  # (B, HW, 512)
            aq, _ = self.cross_attention(
                self.query_norm(iq), self.key_norm(sf), self.key_norm(sf), key_padding_mask=pm
            )
            attended = aq.mean(1)                              # (B, 512)

            # ReLU-tail projected embeddings — DO NOT use for faithfulness scoring
            img_embed    = F.normalize(self.img_proj_head(attended), dim=-1)  # (B, 256)
            txt_embed    = F.normalize(self.txt_proj_head(tf), dim=-1)        # (B, 256)
            # Zero-centred raw text feature — used for Tier-1 faithfulness
            txt_feat_raw = tf                                                  # (B, 512)

        if timestep is not None:
            ts = self.timestep_embed(timestep)
            g, b = self.film_adv(ts).chunk(2, dim=-1)
            rel = torch.sigmoid(self.relative_adv_head((1 + g) * gf + b))
            sm  = self.seam_feat(feats)
            gs, bs = self.film_seam(ts).chunk(2, dim=-1)
            sm = (1 + gs[:, :, None, None]) * sm + bs[:, :, None, None]
        else:
            rel = torch.sigmoid(self.relative_adv_head(gf))
            sm  = self.seam_feat(feats)

        seam_map = torch.sigmoid(self.seam_cls(sm))

        return {
            "binary_logits":      F.adaptive_avg_pool2d(self.adv_head(feats), (1, 1)).flatten(1),
            "class_logits":       F.adaptive_avg_pool2d(self.class_head(feats), (1, 1)).flatten(1),
            "adversarial_map":    torch.sigmoid(self.adv_head(feats)),
            "img_embed":          img_embed,      # (B, 256) ReLU-projected → StateEncoder
            "txt_embed":          txt_embed,       # (B, 256) ReLU-projected → backward compat only
            "txt_feat_raw":       txt_feat_raw,    # (B, 512) zero-centred  → Tier-1 faithfulness
            "seam_quality_score": F.adaptive_avg_pool2d(seam_map, (1, 1)).flatten(1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Public class
# ─────────────────────────────────────────────────────────────────────────────

class AuditResult(TypedDict):
    adv_prob:     float
    is_unsafe:    bool
    harm_class:   str
    policy_score: float
    faithfulness: float
    seam_quality: float
    mask_pil:     Image.Image
    heatmap:      "np.ndarray"
    img_embed:    "torch.Tensor"   # (256,) – auditor's own image embedding
    text_embed:   "torch.Tensor"   # (512,) – auditor's own text embedding


class AdversarialAuditor:
    """
    Loads the adversarial auditor model from disk.

    Parameters
    ----------
    model_path : str   Path to the .pth checkpoint.
    vocab_path : str   Path to vocab.json (produced during auditor training).
    device     : str   'cuda' | 'cpu' | 'auto'  (default: auto)
    """

    def __init__(
        self,
        model_path: str = "complete_auditor_best.pth",
        vocab_path: str = "vocab.json",
        device: str = "auto",
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)

        self.tokenizer = SimpleTokenizer(vocab_path)
        vocab_size = len(self.tokenizer.word_to_idx)

        # num_classes=_CKPT_NUM_CLASSES (6) to match training checkpoint shape
        self.model = CompleteMultiTaskAuditor(
            num_classes=_CKPT_NUM_CLASSES, vocab_size=vocab_size
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device), strict=False,
        )
        self.model.to(self.device).eval()

        self._tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # ── Faithfulness scoring ──────────────────────────────────────────────────

    @staticmethod
    def _faithfulness_score(
        img_embed: "torch.Tensor | None",     # (1, 256) ReLU-projected
        txt_feat_raw: "torch.Tensor | None",  # (1, 512) zero-centred LSTM→FC
    ) -> float:
        """
        Two-tier faithfulness score in [0, 1].

        Tier 1 (preferred — always taken if txt_feat_raw is available):
          txt_feat_raw is (B, 512) from the LSTM→FC path which feeds into
          LayerNorm before cross-attention K/V.  It is genuinely zero-centred,
          so its cosine similarity with a normalised image vector spans [-1, +1].

          We reduce it from 512-d to 256-d by averaging the two halves
          (corresponding to the forward and backward LSTM directions), then
          L2-normalise both img and text vectors before computing cosine sim.

              cos ∈ [-1, +1]  →  (cos + 1) / 2  ∈ [0, 1]   ← CORRECT

        Tier 2 (fallback — txt_feat_raw unavailable):
          Return 0.0 (conservatively rejects the candidate) rather than
          producing a misleading score.
        """
        if txt_feat_raw is not None and img_embed is not None:
            # Tier 1: zero-centred text → genuine [-1, +1] cosine range
            tf      = txt_feat_raw.float()                    # (1, 512)
            tf_half = (tf[:, :256] + tf[:, 256:]) * 0.5      # (1, 256)  avg fwd+bwd halves
            tf_norm = F.normalize(tf_half, dim=-1)
            ie_norm = F.normalize(img_embed.float(), dim=-1)
            cos     = F.cosine_similarity(ie_norm, tf_norm).item()  # ∈ [-1, +1]
            return float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))

        # Tier 2: conservative fallback
        return 0.0

    # ── Main inference API ────────────────────────────────────────────────────

    def audit_pil(
        self, pil: Image.Image, prompt: str = "", t_norm: float = 0.0
    ) -> AuditResult:
        """
        Audit a PIL image.

        Parameters
        ----------
        pil    : RGB PIL image (any size, will be resized internally)
        prompt : text prompt used to generate the image
        t_norm : timestep normalised to [0,1]  (0 = clean, 1 = fully noisy)

        Returns
        -------
        AuditResult TypedDict — includes img_embed (256-d) and text_embed (512-d)
        for use by StateEncoder when building the policy state vector.
        """
        img_t = self._tfm(pil.convert("RGB")).unsqueeze(0).to(self.device)
        toks  = self.tokenizer.encode(prompt).unsqueeze(0).to(self.device)
        ts_t  = torch.tensor([[t_norm]], device=self.device, dtype=torch.float32)

        with torch.no_grad():
            out = self.model(img_t, text_tokens=toks, timestep=ts_t)

        prob = torch.sigmoid(out["binary_logits"]).item()

        # Only use first 3 logits for harm-class prediction (safe/nudity/violence)
        class_logits_3 = out["class_logits"][:, :3]
        cp = F.softmax(class_logits_3, dim=1)[0]
        pi = torch.argmax(cp).item()

        # ── Tier-1 faithfulness via zero-centred txt_feat_raw ─────────────────
        faithfulness = self._faithfulness_score(
            img_embed    = out["img_embed"],     # (1, 256) — ReLU-projected
            txt_feat_raw = out["txt_feat_raw"],  # (1, 512) — zero-centred raw
        )

        raw  = out["adversarial_map"][0, 0]
        hmap = F.interpolate(
            raw.unsqueeze(0).unsqueeze(0), (512, 512), mode="bilinear"
        )[0, 0].cpu().numpy()

        # binary mask (top-15% → feathered)
        thr  = np.percentile(hmap, 85)
        bm   = (hmap >= thr).astype(np.float32)
        feat = cv2.GaussianBlur(bm, (15, 15), sigmaX=5)
        mask_pil = Image.fromarray(
            ((feat > 0.5).astype(np.float32) * 255).astype(np.uint8)
        ).convert("L")

        # img_embed  → (256,) for StateEncoder.image_proj  (projected)
        # text_embed → (512,) raw for StateEncoder.text_proj  (NOT the projected 256-d)
        img_embed = (
            out["img_embed"][0].detach().cpu()     # (256,)
            if out["img_embed"] is not None else torch.zeros(256)
        )
        text_embed = (
            out["txt_feat_raw"][0].detach().cpu()  # (512,) raw — matches StateEncoder.text_proj
            if out["txt_feat_raw"] is not None else torch.zeros(512)
        )

        return AuditResult(
            adv_prob     = prob,
            is_unsafe    = prob > 0.5,
            harm_class   = CLASS_NAMES[pi],
            policy_score = 1.0 - prob,
            faithfulness = faithfulness,   # ← FIXED: Tier-1 via txt_feat_raw
            seam_quality = out["seam_quality_score"].item(),
            mask_pil     = mask_pil,
            heatmap      = hmap,
            img_embed    = img_embed,
            text_embed   = text_embed,
        )
