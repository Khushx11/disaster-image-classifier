"""Frozen visual inference and training-only fitted text representations."""

import os, json, time, joblib, hashlib
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, ImageOps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from .data import ROOT


class Images(Dataset):
    def __init__(self, frame, transform):
        self.paths = frame.image_path.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(ROOT / "data/raw" / self.paths[i]) as image:
            return self.transform(ImageOps.exif_transpose(image).convert("RGB"))


def image_cache_identity(frame, raw_directory):
    """Fingerprint ordered image bytes, preprocessing and relevant library versions."""
    digest = hashlib.sha256()
    for path in frame.image_path:
        encoded = str(path).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        with (Path(raw_directory) / path).open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    return {
        "schema": 1,
        "ordered_image_sha256": digest.hexdigest(),
        "records": len(frame),
        "dimension": 512,
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "transforms": str(ResNet18_Weights.IMAGENET1K_V1.transforms()),
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "torchvision", "Pillow"]
        },
    }


def image_cache_matches(cache, identity):
    cache = Path(cache)
    try:
        recorded = json.loads((cache / "image_identity.json").read_text())
        if recorded != identity or not (cache / "image_metadata.json").is_file():
            return False
        array = np.load(cache / "image.npy", mmap_mode="r", allow_pickle=False)
        return (
            array.shape == (identity["records"], identity["dimension"])
            and array.dtype == np.float32
        )
    except (OSError, ValueError, KeyError):
        return False


def extract():
    torch.set_num_threads(4)
    frame = pd.read_csv(ROOT / "data/manifest.csv").fillna({"text": ""})
    cache = ROOT / "work/features"
    cache.mkdir(parents=True, exist_ok=True)
    train = (frame.split == "train").to_numpy()
    identity = image_cache_identity(frame, ROOT / "data/raw")
    if not image_cache_matches(cache, identity):
        print(
            "Building visual cache: inputs or encoder identity are new or changed.",
            flush=True,
        )
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval().to(device)
        loader = DataLoader(
            Images(frame, weights.transforms()), batch_size=64, num_workers=0
        )
        parts = []
        start = time.perf_counter()
        with torch.inference_mode():
            for batch, b in enumerate(loader):
                parts.append(model(b.to(device)).cpu().numpy())
                if batch % 25 == 0:
                    print(
                        f"Visual features: {min((batch+1)*64,len(frame))}/{len(frame)}, {time.perf_counter()-start:.1f}s",
                        flush=True,
                    )
        np.save(cache / "image.npy", np.concatenate(parts).astype("float32"))
        (cache / "image_metadata.json").write_text(
            json.dumps(
                {
                    "device": device,
                    "seconds": time.perf_counter() - start,
                    "weights": "ResNet18_Weights.IMAGENET1K_V1",
                    "torch": torch.__version__,
                },
                indent=2,
            )
        )
        (cache / "image_identity.json").write_text(json.dumps(identity, indent=2))
    else:
        print(
            "Reusing verified visual cache: ordered inputs and encoder identity match.",
            flush=True,
        )
    vectorizer = TfidfVectorizer(
        max_features=20000, min_df=2, ngram_range=(1, 2), sublinear_tf=True
    )
    sparse_train = vectorizer.fit_transform(frame.loc[train, "text"])
    svd = TruncatedSVD(n_components=256, random_state=42)
    svd.fit(sparse_train)
    text = (
        Normalizer()
        .fit_transform(svd.transform(vectorizer.transform(frame.text)))
        .astype("float32")
    )
    image = Normalizer().fit_transform(np.load(cache / "image.npy")).astype("float32")
    np.savez_compressed(
        cache / "features.npz",
        text=text,
        image=image,
        y=frame.y.to_numpy(),
        split=frame.split.to_numpy(),
        group=frame.group.to_numpy(),
        manifest_sha256=np.array(
            hashlib.sha256((ROOT / "data/manifest.csv").read_bytes()).hexdigest()
        ),
    )
    joblib.dump(
        {"vectorizer": vectorizer, "svd": svd}, ROOT / "results/text_pipeline.joblib"
    )
    (ROOT / "results/feature_metadata.json").write_text(
        json.dumps(
            {
                "text_vocab": len(vectorizer.vocabulary_),
                "svd_explained_variance": float(svd.explained_variance_ratio_.sum()),
                "text_dimension": text.shape[1],
                "image_dimension": image.shape[1],
                "fit_split": "train",
                "manifest_sha256": hashlib.sha256(
                    (ROOT / "data/manifest.csv").read_bytes()
                ).hexdigest(),
                "image_cache_identity": identity,
                "image_extraction": json.loads(
                    (cache / "image_metadata.json").read_text()
                ),
            },
            indent=2,
        )
    )
    print("Feature extraction complete", flush=True)


def validate_feature_rows(features, root):
    """Refuse feature/label partitions that no longer match the current manifest."""
    root = Path(root)
    frame = pd.read_csv(root / "data/manifest.csv")
    for key in ["y", "split", "group"]:
        if not np.array_equal(features[key], frame[key].to_numpy()):
            raise ValueError(
                f"Cached {key} rows differ from the manifest; rerun features in a fresh reproduction."
            )
    if "manifest_sha256" in features:
        digest = hashlib.sha256((root / "data/manifest.csv").read_bytes()).hexdigest()
        if str(features["manifest_sha256"].item()) != digest:
            raise ValueError(
                "Feature cache belongs to a different manifest; rerun features."
            )
    if len(features["text"]) != len(frame) or len(features["image"]) != len(frame):
        raise ValueError("Feature and manifest record counts differ.")
