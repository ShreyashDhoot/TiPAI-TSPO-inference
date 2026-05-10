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
adv_prob      float   P(adversarial)  ∈ [0,1]
is_unsafe     bool    adv_prob > 0.5
harm_class    str     'safe' | 'nudity' | 'violence'
policy_score  float   1 - adv_prob
faithfulness  float   cos-similarity rescaled to [0,1]
seam_quality  float   seam/blend quality score
mask_pil      PIL.Image  binary inpaint mask (L mode, 512×512)
heatmap       np.ndarray  raw adversarial heatmap (512×512, float32)
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


# ─────────────────────────────────────────────────────────────────────────────
# Internal sub-models
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
        return self.fc(h), self.norm(self.fc(out)), toks.eq(0)


class CompleteMultiTaskAuditor(nn.Module):
    def __init__(self, num_classes: int = 3, vocab_size: int = 50_000):
        super().__init__()
        resnet = models.resnet101(weights=None)
        self.features   = nn.Sequential(*list(resnet.children())[:-2])
        self.text_encoder = SimpleTextEncoder(vocab_size=vocab_size)
        self.adv_head   = nn.Conv2d(2048, 1, 1)
        self.class_head = nn.Conv2d(2048, num_classes, 1)
        self.quality_head = nn.Conv2d(2048, 1, 1)
        self.object_detection_head = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, num_classes, 1)
        )
        self.image_proj = nn.Conv2d(2048, 512, 1)
        self.cross_attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        self.query_norm = nn.LayerNorm(512)
        self.key_norm   = nn.LayerNorm(512)
        self.img_proj_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 256))
        self.txt_proj_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 256))
        self.log_temperature = nn.Parameter(torch.tensor([-2.659]))
        self.timestep_embed = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 512)
        )
        self.film_adv  = nn.Linear(512, 2048 * 2)
        self.film_seam = nn.Linear(512, 512 * 2)
        self.relative_adv_head = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1)
        )
        self.seam_feat = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512)
        )
        self.seam_cls = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256), nn.Conv2d(256, 1, 1)
        )

    def forward(self, x: torch.Tensor, text_tokens=None, timestep=None):
        B = x.size(0)
        feats = self.features(x)
        gf = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)

        img_embed = txt_embed = None
        if text_tokens is not None:
            tf, sf, pm = self.text_encoder(text_tokens)
            iq = self.image_proj(feats).view(B, 512, -1).permute(0, 2, 1)
            aq, _ = self.cross_attention(
                self.query_norm(iq), self.key_norm(sf), self.key_norm(sf), key_padding_mask=pm
            )
            img_embed = F.normalize(self.img_proj_head(aq.mean(1)), dim=-1)
            txt_embed = F.normalize(self.txt_proj_head(tf), dim=-1)

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
            "img_embed":          img_embed,
            "txt_embed":          txt_embed,
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
    heatmap:      np.ndarray


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

        self.model = CompleteMultiTaskAuditor(3, vocab_size)
        state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model_state_dict", state))
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self._tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
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
        AuditResult TypedDict
        """
        img_t = self._tfm(pil.convert("RGB")).unsqueeze(0).to(self.device)
        toks  = self.tokenizer.encode(prompt).unsqueeze(0).to(self.device)
        ts_t  = torch.tensor([[t_norm]], device=self.device, dtype=torch.float32)

        out = self.model(img_t, text_tokens=toks, timestep=ts_t)

        prob  = torch.sigmoid(out["binary_logits"]).item()
        cp    = F.softmax(out["class_logits"], dim=1)[0]
        pi    = torch.argmax(cp).item()
        cos   = (
            F.cosine_similarity(out["img_embed"], out["txt_embed"]).item()
            if out["img_embed"] is not None else 0.0
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

        return AuditResult(
            adv_prob     = prob,
            is_unsafe    = prob > 0.5,
            harm_class   = CLASS_NAMES[pi],
            policy_score = 1.0 - prob,
            faithfulness = (cos + 1.0) / 2.0,
            seam_quality = out["seam_quality_score"].item(),
            mask_pil     = mask_pil,
            heatmap      = hmap,
        )
