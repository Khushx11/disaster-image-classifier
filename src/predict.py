"""Run inference with the capstone's selected, trusted local checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.preprocessing import normalize
import torch
from torchvision.models import resnet18, ResNet18_Weights

from .data import ROOT, clean_text
from .models import build_model

CLASSES = ("not_informative", "informative")


class Predictor:
    def __init__(self, root=ROOT, *, compare=False):
        torch.set_num_threads(4)
        results = root / "results"
        lock = json.loads((results / "models/selection_locked.json").read_text())
        self.protocol = lock["protocol"]
        if self.protocol != json.loads((root / "protocol.json").read_text()):
            raise ValueError("Checkpoint selection and experiment protocol differ.")
        self.models = {}
        for kind in ("gated", "image", "text") if compare else ("gated",):
            models = []
            for seed in self.protocol["seeds"]:
                name = f"{kind}_{seed}"
                checkpoint = results / "models" / name / "best.pt"
                with checkpoint.open("rb") as file:
                    digest = hashlib.file_digest(file, "sha256").hexdigest()
                if digest != lock["checkpoint_sha256"][name]:
                    raise ValueError(f"Checkpoint checksum mismatch: {name}")
                model = build_model(kind, self.protocol)
                model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
                models.append(model.eval())
            self.models[kind] = models

        self.text_pipeline = joblib.load(results / "text_pipeline.joblib")
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.transform = weights.transforms()
        self.encoder = resnet18(weights=weights)
        self.encoder.fc = torch.nn.Identity()
        self.encoder.eval()

    @torch.inference_mode()
    def run(self, image_source, text=None):
        with Image.open(image_source) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"} or image.width * image.height > 20_000_000:
                raise ValueError("Use a JPEG, PNG or WebP image no larger than 20 megapixels.")
            image = ImageOps.exif_transpose(image).convert("RGB")
            visual = self.transform(image).unsqueeze(0)
        visual = torch.from_numpy(normalize(self.encoder(visual).numpy()).astype("float32"))
        words = clean_text(text or "")
        sparse = self.text_pipeline["vectorizer"].transform([words])
        recognized = sparse.nnz > 0
        text_status = "used" if recognized else "unrecognized" if words else "absent"
        projected = self.text_pipeline["svd"].transform(sparse)
        text_features = torch.from_numpy(normalize(projected).astype("float32"))
        available = torch.tensor([[float(recognized), 1.0]])
        gated_runs = [model(text_features, visual, available) for model in self.models["gated"]]
        gated = torch.stack([item[0].softmax(1) for item in gated_runs]).mean(0)[0]
        result = {
            "prediction": CLASSES[int(gated.argmax())],
            "probability_informative": float(gated[1]),
            "mean_text_gate_weight": float(np.mean([item[2].item() for item in gated_runs])),
            "text_available": recognized,
            "text_status": text_status,
        }
        if "image" in self.models:
            image_probs = [model(text_features, visual, available)[0].softmax(1) for model in self.models["image"]]
            image_only = torch.stack(image_probs).mean(0)[0]
            late = image_only
            if recognized:
                text_probs = [model(text_features, visual, available)[0].softmax(1) for model in self.models["text"]]
                late = torch.stack([(t + i) / 2 for t, i in zip(text_probs, image_probs, strict=True)]).mean(0)[0]
            result["comparison"] = {
                "late_fusion": float(late[1]), "image_only": float(image_only[1]),
            }
        return result


def predict(image_path, text=None):
    return Predictor().run(image_path, text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--text")
    parser.add_argument("--evidence", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(Predictor(args.evidence).run(args.image, args.text), indent=2))
