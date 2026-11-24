"""Fusion architectures implemented directly, without a fusion-model library."""

import torch
from torch import nn


class Fusion(nn.Module):
    def __init__(self, kind, text_dim=256, image_dim=512, hidden=128, dropout=0.2):
        super().__init__()
        self.kind = kind
        self.hidden = hidden

        def projection(d):
            return nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout))

        self.text = projection(text_dim) if kind != "image" else None
        self.image = projection(image_dim) if kind != "text" else None
        if kind in ["text", "image"]:
            self.classifier = nn.Linear(hidden, 2)
        elif kind in ["concat", "concat_drop"]:
            self.classifier = nn.Sequential(
                nn.Linear(2 * hidden + 2, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 2),
            )
        else:
            self.gate = nn.Linear(4 * hidden + 2, hidden) if kind != "no_gate" else None
            self.classifier = nn.Sequential(
                nn.Linear(3 * hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 2),
            )
            self.text_aux = nn.Linear(hidden, 2)
            self.image_aux = nn.Linear(hidden, 2)

    def forward(self, text, image, available=None):
        if available is None:
            available = torch.ones((len(text), 2), device=text.device)
        mt, mi = available[:, 0:1], available[:, 1:2]
        ht = (
            self.text(text) * mt
            if self.text is not None
            else torch.zeros((len(text), self.hidden), device=text.device)
        )
        hi = (
            self.image(image) * mi
            if self.image is not None
            else torch.zeros((len(text), self.hidden), device=text.device)
        )
        if self.kind == "text":
            return self.classifier(ht), None, None
        if self.kind == "image":
            return self.classifier(hi), None, None
        if self.kind in ["concat", "concat_drop"]:
            return self.classifier(torch.cat([ht, hi, available], 1)), None, None
        difference = (ht - hi).abs() * (mt * mi)
        interaction = ht * hi
        g = (
            torch.sigmoid(
                self.gate(torch.cat([ht, hi, difference, interaction, available], 1))
            )
            if self.gate is not None
            else torch.full_like(ht, 0.5)
        )
        denominator = (mt * g + mi * (1 - g)).clamp_min(1e-8)
        wt = mt * g / denominator
        wi = mi * (1 - g) / denominator
        z = wt * ht + wi * hi
        logits = self.classifier(torch.cat([z, difference, interaction], 1))
        return logits, (self.text_aux(ht), self.image_aux(hi)), wt.mean(1)


def build_model(kind, protocol, text_dim=256, image_dim=512):
    """Use the same architecture settings for training, evaluation and the demo."""
    if kind not in {
        "text",
        "image",
        "concat",
        "concat_drop",
        "gated",
        "no_gate",
        "no_aux",
        "no_drop",
    }:
        raise ValueError(f"Unknown fusion model: {kind}")
    cfg = protocol["training"]
    return Fusion(
        kind,
        text_dim=text_dim,
        image_dim=image_dim,
        hidden=int(cfg["hidden"]),
        dropout=float(cfg["dropout"]),
    )


def training_availability(size, kind, config, device=None):
    """Drop either modality with equal probability; never drop both."""
    probability = float(config["modality_drop_probability"])
    if not 0 <= probability <= 1:
        raise ValueError("modality_drop_probability must lie in [0,1]")
    available = torch.ones((size, 2), device=device)
    if kind in {"concat_drop", "gated", "no_gate", "no_aux"}:
        draw = torch.rand(size, device=device)
        available[draw < probability / 2, 0] = 0
        available[(draw >= probability / 2) & (draw < probability), 1] = 0
    return available
