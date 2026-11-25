"""Evaluate fixed linear baselines on each entirely held-out CrisisMMD event."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(manifest, image_features, image_manifest):
    alignment = ["event_name", "group", "y", "image_path", "pixel_hash"]
    frame = pd.read_csv(manifest, usecols=[*alignment, "text"])
    cached_rows = pd.read_csv(image_manifest, usecols=alignment)
    if not frame[alignment].equals(cached_rows[alignment]):
        raise ValueError("Image cache manifest does not match the ordered evaluation rows")
    image_manifest_hash = digest(image_manifest)
    with np.load(image_features, allow_pickle=False) as cache:
        if cache["manifest_sha256"].item() != image_manifest_hash:
            raise ValueError("Feature archive was generated from a different manifest")
        if not np.array_equal(cache["y"], cached_rows.y.to_numpy()):
            raise ValueError("Feature labels do not match the ordered image manifest")
        images = cache["image"]
    if images.shape != (len(frame), 512) or not np.isfinite(images).all():
        raise ValueError("Expected one finite 512-dimensional ResNet18 vector per row")
    if frame[["event_name", "group", "y"]].isna().any().any() or set(frame.y) != {0, 1}:
        raise ValueError("Expected valid event names, duplicate groups and binary image labels")
    frame["text"] = frame.text.fillna("")
    results = []
    for event in sorted(frame.event_name.unique()):
        test = frame.loc[frame.event_name == event]
        other = frame.loc[frame.event_name != event]
        train = other.loc[~other.group.isin(test.group)]
        if set(train.y) != {0, 1} or set(test.y) != {0, 1}:
            raise ValueError(f"Expected both classes in training and test for {event}")

        # Fit all learned transforms on the fold's training rows.
        text_model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30_000),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        )
        image_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        )
        with threadpool_limits(limits=4):
            text_model.fit(train.text, train.y)
            image_model.fit(images[train.index], train.y)
            text_score = text_model.predict_proba(test.text)[:, 1]
            image_score = image_model.predict_proba(images[test.index])[:, 1]
        predictions = {
            "text": text_score > 0.5,
            "image": image_score > 0.5,
            "late_fusion": (image_score + text_score) / 2 > 0.5,
            "majority": [int(train.y.mean() > 0.5)] * len(test),
        }
        results.append({
            "event": event,
            "training_rows": len(train),
            "cross_event_group_rows_removed": len(other) - len(train),
            "test_rows": len(test),
            "test_label_counts": {str(label): int((test.y == label).sum()) for label in (0, 1)},
            "metrics": {
                name: {
                    "macro_f1": f1_score(test.y, prediction, labels=[0, 1], average="macro"),
                    "confusion_matrix": confusion_matrix(test.y, prediction, labels=[0, 1]).tolist(),
                }
                for name, prediction in predictions.items()
            },
        })
    return {
        "protocol": "Leave one event out; purge duplicate groups shared with the held-out event.",
        "target": "Original CrisisMMD image-informativeness label.",
        "models": {
            "text": "Word unigram/bigram TF-IDF and balanced logistic regression (C=1).",
            "image": "L2-normalised frozen ImageNet ResNet18 features, standardisation and balanced logistic regression (C=1).",
            "late_fusion": "Equal average of the image and text scores, thresholded at 0.5.",
            "majority": "Most frequent class in the fold's training rows.",
        },
        "manifest_sha256": digest(manifest),
        "image_manifest_sha256": image_manifest_hash,
        "image_features_sha256": digest(image_features),
        "scikit_learn_version": sklearn.__version__,
        "event_mean_macro_f1": {
            name: sum(row["metrics"][name]["macro_f1"] for row in results) / len(results)
            for name in results[0]["metrics"]
        },
        "events": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.manifest, args.image_features, args.image_manifest)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
