"""Meaningful invariants and result consistency checks, without retraining."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import ROOT
from .models import Fusion, build_model, training_availability


def run():
    torch.manual_seed(9)
    t = torch.randn(7, 256)
    i = torch.randn(7, 512)
    for kind in ["gated", "no_gate", "no_aux", "no_drop", "concat_drop"]:
        model = Fusion(kind).eval()
        with torch.inference_mode():
            for absent in [0, 1]:
                a = torch.ones(7, 2)
                a[:, absent] = 0
                original = model(t, i, a)[0]
                changed = model(
                    t * 999 if absent == 0 else t, i * 999 if absent == 1 else i, a
                )[0]
                assert torch.allclose(
                    original, changed, atol=1e-6
                ), f"Missing-input invariance: {kind}"
            assert torch.isfinite(model(t, i)[0]).all()
        logits, aux, _ = model(t, i)
        loss = logits.square().mean()
        loss.backward()
        assert model.classifier[-1].weight.grad.abs().sum() > 0
    protocol = json.loads((ROOT / "protocol.json").read_text())
    changed = json.loads(json.dumps(protocol))
    changed["training"].update(hidden=17, dropout=0.35)
    configured = build_model("gated", changed)
    assert configured.hidden == 17 and configured.text[2].p == 0.35
    assert configured(t, i)[0].shape == (7, 2)
    for probability in [0.0, 1.0]:
        flags = training_availability(
            100, "gated", {"modality_drop_probability": probability}
        )
        assert torch.all(flags.sum(1) == (2 if probability == 0 else 1))
    from tempfile import TemporaryDirectory
    from PIL import Image
    from .features import image_cache_identity, image_cache_matches

    with TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for name, value in [("a.png", 0), ("b.png", 255)]:
            Image.new("RGB", (4, 4), (value, value, value)).save(temporary / name)
        images = pd.DataFrame({"image_path": ["a.png", "b.png"]})
        identity = image_cache_identity(images, temporary)
        assert identity != image_cache_identity(images.iloc[::-1], temporary)
        np.save(temporary / "image.npy", np.zeros((2, 512), dtype="float32"))
        (temporary / "image_identity.json").write_text(json.dumps(identity))
        (temporary / "image_metadata.json").write_text("{}")
        assert image_cache_matches(temporary, identity)
        Image.new("RGB", (4, 4), (127, 127, 127)).save(temporary / "a.png")
        assert not image_cache_matches(
            temporary, image_cache_identity(images, temporary)
        )
    manifest = ROOT / "data/manifest.csv"
    if manifest.exists():
        frame = pd.read_csv(manifest)
        for col in ["group", "tweet_id", "pixel_hash"]:
            assert (
                frame.groupby(col).split.nunique().max() == 1
            ), f"Split leakage in {col}"
        assert set(frame.y) == {0, 1}
        assert np.array_equal(
            frame.y.to_numpy(), (frame.label == "informative").astype(int).to_numpy()
        )
    cache = ROOT / "work/features/features.npz"
    if cache.exists():
        cached = np.load(cache, allow_pickle=True)
        assert np.array_equal(cached["y"], frame.y.to_numpy())
        assert np.array_equal(cached["split"], frame.split.to_numpy())
    if (ROOT / "results/metrics.csv").exists():
        from .evaluate import metrics

        table = pd.read_csv(ROOT / "results/metrics.csv")
        for file in (ROOT / "results/predictions").glob("*.npz"):
            kind, seed = file.stem.rsplit("_", 1)
            data = np.load(file)
            for condition in ["clean", "no_text", "no_image", "shuffled_text"]:
                assert np.allclose(data[condition].sum(1), 1, atol=1e-5)
                row = table[
                    (table.model == kind)
                    & (table.seed == int(seed))
                    & (table.condition == condition)
                ].iloc[0]
                for key, value in metrics(data["y"], data[condition]).items():
                    assert abs(value - row[key]) < 1e-7, (file, key)
    if (ROOT / "results/predictions/image_42.npz").exists():
        visual = np.load(ROOT / "results/predictions/image_42.npz")
        assert np.array_equal(visual["clean"], visual["no_text"])
        assert np.array_equal(visual["clean"], visual["shuffled_text"])
    print(
        "Passed missing-input invariance, gradient flow, configuration and cache-identity checks, split checks and saved-metric consistency."
    )
