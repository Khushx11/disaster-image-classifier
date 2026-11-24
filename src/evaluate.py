"""Held-out metrics, robustness and group-aware paired inference."""

import json, hashlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
)
from .data import ROOT
from .features import validate_feature_rows
from .models import build_model


def metrics(y, p):
    pred = p.argmax(1)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
        "auroc": roc_auc_score(y, p[:, 1]),
        "average_precision": average_precision_score(y, p[:, 1]),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": float(np.mean((p[:, 1] - y) ** 2)),
    }


def f1_cm(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    den = cm.sum(-1) + cm.sum(-2)
    return np.divide(
        2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0
    ).mean(-1)


def group_statistics(y, a, b, groups, repetitions=10000, seed=2026):
    unique, ids = np.unique(groups, return_inverse=True)
    ca = np.zeros((len(unique), 2, 2))
    cb = ca.copy()
    np.add.at(ca, (ids, y, a.argmax(1)), 1)
    np.add.at(cb, (ids, y, b.argmax(1)), 1)
    observed = float(f1_cm(ca.sum(0)) - f1_cm(cb.sum(0)))
    rng = np.random.default_rng(seed)
    boot = []
    null = []
    for start in range(0, repetitions, 100):
        n = min(100, repetitions - start)
        counts = rng.multinomial(
            len(unique), np.full(len(unique), 1 / len(unique)), size=n
        )
        boot.extend(
            (
                f1_cm(np.einsum("ng,gij->nij", counts, ca))
                - f1_cm(np.einsum("ng,gij->nij", counts, cb))
            ).tolist()
        )
        swap = rng.integers(0, 2, size=(n, len(unique)))
        perm_a = ca.sum(0)[None] + np.einsum("ng,gij->nij", swap, cb - ca)
        perm_b = cb.sum(0)[None] + np.einsum("ng,gij->nij", swap, ca - cb)
        null.extend((f1_cm(perm_a) - f1_cm(perm_b)).tolist())
    return {
        "difference_macro_f1": observed,
        "bootstrap_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "permutation_p_two_sided": float(
            (1 + np.sum(np.abs(null) >= abs(observed))) / (repetitions + 1)
        ),
        "groups": len(unique),
        "repetitions": repetitions,
    }


def run():
    torch.set_num_threads(4)
    assert (
        ROOT / "results/models/selection_locked.json"
    ).exists(), "Complete and lock model selection first"
    lock = json.loads((ROOT / "results/models/selection_locked.json").read_text())
    for name, digest in lock["checkpoint_sha256"].items():
        assert (
            hashlib.sha256(
                (ROOT / f"results/models/{name}/best.pt").read_bytes()
            ).hexdigest()
            == digest
        ), "Selected checkpoint changed"
    cfg = json.loads((ROOT / "protocol.json").read_text())
    assert cfg == lock["protocol"], "Protocol differs from locked model selection"
    data = np.load(ROOT / "work/features/features.npz", allow_pickle=True)
    validate_feature_rows(data, ROOT)
    ids = np.where(data["split"] == "test")[0]
    y = data["y"][ids]
    text = torch.tensor(data["text"][ids])
    image = torch.tensor(data["image"][ids])
    shuffled = text[torch.tensor(np.random.default_rng(2026).permutation(len(ids)))]
    out = ROOT / "results/predictions"
    out.mkdir(exist_ok=True)
    records = []
    saved = {}
    for kind in cfg["models"]:
        for seed in cfg["seeds"]:
            name = f"{kind}_{seed}"
            model = build_model(
                kind, cfg, text_dim=text.shape[1], image_dim=image.shape[1]
            )
            model.load_state_dict(
                torch.load(ROOT / f"results/models/{name}/best.pt", weights_only=True)
            )
            model.eval()
            probs = {}
            gates = {}
            for condition in cfg["conditions"]:
                avail = torch.ones((len(ids), 2))
                if condition == "no_text":
                    avail[:, 0] = 0
                if condition == "no_image":
                    avail[:, 1] = 0
                with torch.inference_mode():
                    logits, _, gate = model(
                        shuffled if condition == "shuffled_text" else text, image, avail
                    )
                    p = logits.softmax(1).numpy()
                probs[condition] = p
                if gate is not None:
                    gates[condition] = gate.numpy()
                records.append(
                    {
                        "model": kind,
                        "seed": seed,
                        "condition": condition,
                        **metrics(y, p),
                    }
                )
            np.savez_compressed(
                out / f"{name}.npz",
                y=y,
                indices=ids,
                **probs,
                **{"gate_" + k: v for k, v in gates.items()},
            )
            saved[(kind, seed)] = probs
            # Retrospective curves cannot affect the checkpoint already selected.
            epoch_archive = ROOT / f"work/epochs/{name}.pt"
            if epoch_archive.exists():
                states = torch.load(epoch_archive, weights_only=True)
                curves = []
                for epoch, state in enumerate(states, 1):
                    model.load_state_dict(state)
                    with torch.inference_mode():
                        p = model(text, image)[0].softmax(1).numpy()
                    curves.append({"epoch": epoch, **metrics(y, p)})
                (ROOT / f"results/models/{name}/test_curve.json").write_text(
                    json.dumps(curves, indent=2)
                )
            else:
                print(
                    f"{name}: selected-checkpoint metrics evaluated; temporary epoch states absent, existing retrospective curve retained if available.",
                    flush=True,
                )
    for seed in cfg["seeds"]:
        probs = {}
        for condition in cfg["conditions"]:
            t = saved[("text", seed)][condition]
            i = saved[("image", seed)][condition]
            p = (
                i
                if condition == "no_text"
                else t if condition == "no_image" else (t + i) / 2
            )
            probs[condition] = p
            records.append(
                {
                    "model": "late_fusion",
                    "seed": seed,
                    "condition": condition,
                    **metrics(y, p),
                }
            )
        saved[("late_fusion", seed)] = probs
        np.savez_compressed(out / f"late_fusion_{seed}.npz", y=y, indices=ids, **probs)
    pd.DataFrame(records).to_csv(ROOT / "results/metrics.csv", index=False)
    ensemble_records = []
    for kind in cfg["models"] + ["late_fusion"]:
        for condition in cfg["conditions"]:
            p = np.mean([saved[(kind, s)][condition] for s in cfg["seeds"]], 0)
            ensemble_records.append(
                {"model": kind, "condition": condition, **metrics(y, p)}
            )
    pd.DataFrame(ensemble_records).to_csv(
        ROOT / "results/ensemble_metrics.csv", index=False
    )
    stats = {}
    for condition in cfg["conditions"]:
        a = np.mean([saved[("gated", s)][condition] for s in cfg["seeds"]], 0)
        b = np.mean([saved[("concat_drop", s)][condition] for s in cfg["seeds"]], 0)
        stats[condition] = group_statistics(y, a, b, data["group"][ids])
    secondary = cfg["conditions"][1:]
    ordered = sorted(secondary, key=lambda c: stats[c]["permutation_p_two_sided"])
    previous = 0
    for rank, condition in enumerate(ordered):
        adjusted = max(
            previous,
            min(1, (len(ordered) - rank) * stats[condition]["permutation_p_two_sided"]),
        )
        previous = adjusted
        stats[condition]["secondary_holm_p"] = adjusted
    (ROOT / "results/statistics.json").write_text(json.dumps(stats, indent=2))
    print(
        pd.DataFrame(records)
        .query("condition=='clean'")
        .groupby("model")
        .macro_f1.agg(["mean", "std"])
        .to_string(),
        flush=True,
    )
