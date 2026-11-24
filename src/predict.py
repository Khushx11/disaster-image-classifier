"""Single-image demonstration using the submitted custom seed ensemble."""

import argparse, json
from pathlib import Path
import joblib, numpy as np, torch
from PIL import Image, ImageOps
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.preprocessing import normalize
from .models import build_model
from .data import ROOT, clean_text


def predict(image_path, text=None):
    torch.set_num_threads(4)
    weights = ResNet18_Weights.IMAGENET1K_V1
    encoder = resnet18(weights=weights)
    encoder.fc = torch.nn.Identity()
    encoder.eval()
    with Image.open(image_path) as im:
        x = weights.transforms()(ImageOps.exif_transpose(im).convert("RGB")).unsqueeze(
            0
        )
    with torch.inference_mode():
        image = torch.tensor(normalize(encoder(x).numpy()), dtype=torch.float32)
    pipeline = joblib.load(ROOT / "results/text_pipeline.joblib")
    words = clean_text(text or "")
    features = pipeline["svd"].transform(pipeline["vectorizer"].transform([words]))
    t = torch.tensor(normalize(features).astype("float32"))
    available = torch.tensor([[float(bool(words)), 1.0]])
    outputs = []
    gates = []
    protocol = json.loads((ROOT / "results/models/selection_locked.json").read_text())[
        "protocol"
    ]
    for seed in protocol["seeds"]:
        model = build_model(
            "gated", protocol, text_dim=t.shape[1], image_dim=image.shape[1]
        )
        model.load_state_dict(
            torch.load(ROOT / f"results/models/gated_{seed}/best.pt", weights_only=True)
        )
        model.eval()
        with torch.inference_mode():
            logits, _, gate = model(t, image, available)
            outputs.append(logits.softmax(1).numpy()[0])
            gates.append(gate.item())
    probability = np.mean(outputs, 0)
    return {
        "prediction": ["not_informative", "informative"][int(probability.argmax())],
        "probability_informative": float(probability[1]),
        "mean_text_gate_weight": float(np.mean(gates)),
        "text_available": bool(words),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", default=None)
    args = parser.parse_args()
    print(json.dumps(predict(args.image, args.text), indent=2))
