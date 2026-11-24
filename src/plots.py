"""Publication-size monochrome figures generated entirely from saved measurements."""

import json, textwrap
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.decomposition import PCA
from .data import ROOT

NAMES = {
    "text": "Text MLP",
    "image": "Image MLP",
    "concat": "Concatenation",
    "concat_drop": "Concat + dropout",
    "late_fusion": "Late fusion",
    "gated": "Custom gated fusion",
    "no_gate": "Without gate",
    "no_aux": "Without auxiliary loss",
    "no_drop": "Without modality dropout",
    "scratch_cnn": "Scratch CNN",
}
BASELINES = ["text", "image", "concat", "concat_drop", "late_fusion", "gated"]


def run():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "black",
            "text.color": "black",
            "axes.edgecolor": "black",
            "figure.dpi": 160,
            "savefig.facecolor": "white",
            "axes.prop_cycle": matplotlib.cycler(
                color=["black", ".35", ".60", ".15", ".75"]
            ),
        }
    )
    out = ROOT / "results/figures"
    out.mkdir(exist_ok=True)

    def save(name):
        plt.savefig(out / f"{name}.png", dpi=190, bbox_inches="tight")
        plt.close()

    frame = pd.read_csv(ROOT / "data/manifest.csv").fillna({"text": ""})
    scores = pd.read_csv(ROOT / "results/metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    counts = pd.crosstab(frame.event_name.str.replace("_", " "), frame.label)
    counts.plot.barh(ax=axes[0], color=[".25", ".8"], edgecolor="black")
    axes[0].set_xlabel("Image-text records")
    axes[0].set_ylabel("")
    axes[0].legend(["Informative", "Not informative"], fontsize=10)
    for label, style in [(0, "-"), (1, "--")]:
        lengths = frame.loc[frame.y == label, "text"].str.split().str.len()
        axes[1].hist(
            lengths,
            bins=np.arange(0, 65, 3),
            histtype="step",
            color="black",
            linestyle=style,
            label="Informative" if label else "Not informative",
        )
    axes[1].set_xlabel("Cleaned text length (words)")
    axes[1].set_ylabel("Records")
    axes[1].legend(fontsize=10)
    axes[2].hist(
        np.log2(frame.width / frame.height), bins=40, color=".65", edgecolor="black"
    )
    axes[2].set_xlabel("Image aspect ratio, log2(width / height)")
    axes[2].set_ylabel("Records")
    fig.tight_layout()
    save("dataset_distributions")
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    chosen = np.random.default_rng(42).choice(len(frame), 6, replace=False)
    for ax, index in zip(axes.flat, chosen):
        row = frame.iloc[index]
        with Image.open(ROOT / "data/raw" / row.image_path) as im:
            ax.imshow(ImageOps.exif_transpose(im).convert("RGB"))
        ax.axis("off")
        ax.set_title(row.label.replace("_", " "), fontsize=9)
        ax.text(
            0.0,
            -0.04,
            "\n".join(
                textwrap.wrap(row.text.encode("ascii", errors="ignore").decode(), 48)[
                    :3
                ]
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=10,
        )
    fig.subplots_adjust(hspace=0.7)
    save("paired_examples")
    # Label-independent PCA fitted on training features only.
    features = np.load(ROOT / "work/features/features.npz", allow_pickle=True)
    train = features["split"] == "train"
    test = features["split"] == "test"
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key in zip(axes, ["text", "image"]):
        pca = PCA(2, random_state=42).fit(features[key][train])
        xy = pca.transform(features[key][test])
        for label, marker, c in [(0, "x", ".65"), (1, "o", "black")]:
            keep = features["y"][test] == label
            ax.scatter(
                xy[keep, 0],
                xy[keep, 1],
                s=5,
                alpha=0.25,
                marker=marker,
                c=c,
                label="Informative" if label else "Not informative",
            )
        ax.set_title(f"{key.title()} features: held-out records")
        ax.set_xlabel("Training-fitted PC1")
        ax.set_ylabel("PC2")
    axes[1].legend(fontsize=10)
    fig.tight_layout()
    save("feature_distributions")
    clean = scores.query("condition=='clean'")
    summary = clean.groupby("model").agg(
        {
            "macro_f1": ["mean", "std"],
            "accuracy": ["mean", "std"],
            "macro_precision": ["mean", "std"],
            "macro_recall": ["mean", "std"],
            "auroc": ["mean", "std"],
        }
    )
    summary.to_csv(ROOT / "results/summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, keys, title in [
        (axes[0], BASELINES, "Benchmark comparison"),
        (axes[1], ["gated", "no_gate", "no_aux", "no_drop"], "Custom-model ablations"),
    ]:
        means = [clean.loc[clean.model == k, "macro_f1"].mean() for k in keys]
        sd = [clean.loc[clean.model == k, "macro_f1"].std() for k in keys]
        ax.barh(
            range(len(keys)),
            means,
            xerr=sd,
            color=["black" if k == "gated" else ".65" for k in keys],
            edgecolor="black",
            capsize=3,
        )
        ax.set_yticks(range(len(keys)), [NAMES[k] for k in keys])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Test macro-F1 (mean ± seed SD)")
        ax.set_title(title)
    fig.tight_layout()
    save("comparison_ablations")
    fig, ax = plt.subplots(figsize=(10, 3.7))
    conditions = ["clean", "no_text", "no_image", "shuffled_text"]
    x = np.arange(len(conditions))
    width = 0.13
    for j, k in enumerate(BASELINES):
        subset = scores[scores.model == k]
        mean = [
            subset.loc[subset.condition == c, "macro_f1"].mean() for c in conditions
        ]
        sd = [subset.loc[subset.condition == c, "macro_f1"].std() for c in conditions]
        ax.bar(
            x + (j - 2.5) * width,
            mean,
            width,
            yerr=sd,
            color=str(0.12 + j * 0.14),
            edgecolor="black",
            label=NAMES[k],
            hatch=["", "//", "xx", "..", "\\\\", "--"][j],
            capsize=2,
        )
    ax.set_xticks(x, ["Clean", "Text absent", "Image absent", "Text shuffled"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test macro-F1")
    ax.legend(ncol=3, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.24))
    fig.tight_layout()
    save("robustness")
    probs = {
        k: np.mean(
            [
                np.load(ROOT / f"results/predictions/{k}_{s}.npz")["clean"]
                for s in [42, 43, 44]
            ],
            0,
        )
        for k in BASELINES
    }
    ids = np.load(ROOT / "results/predictions/gated_42.npz")["indices"]
    y = frame.y.to_numpy()[ids]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    cm = confusion_matrix(y, probs["gated"].argmax(1), labels=[0, 1])
    axes[0].imshow(cm, cmap="Greys")
    for (a, b), v in np.ndenumerate(cm):
        axes[0].text(
            b,
            a,
            str(v),
            ha="center",
            va="center",
            color="white" if v > cm.max() / 2 else "black",
        )
    axes[0].set_xticks([0, 1], ["Not informative", "Informative"], rotation=15)
    axes[0].set_yticks([0, 1], ["Not informative", "Informative"])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Custom seed ensemble")
    for j, k in enumerate(BASELINES):
        fpr, tpr, _ = roc_curve(y, probs[k][:, 1])
        precision, recall, _ = precision_recall_curve(y, probs[k][:, 1])
        style = ["-", "--", ":", "-.", "-", "--"][j]
        axes[1].plot(fpr, tpr, color=str(j * 0.14), ls=style, label=NAMES[k])
        axes[2].plot(recall, precision, color=str(j * 0.14), ls=style)
    axes[1].plot([0, 1], [0, 1], ":", color=".8")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[2].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    save("classification_diagnostics")
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.2))
    for ax, k in zip(
        axes.flat,
        [
            "text",
            "image",
            "concat",
            "concat_drop",
            "gated",
            "no_gate",
            "no_aux",
            "no_drop",
        ],
    ):
        history = pd.read_json(ROOT / f"results/models/{k}_42/history.json")
        testcurve = pd.read_json(ROOT / f"results/models/{k}_42/test_curve.json")
        metadata = json.loads(
            (ROOT / f"results/models/{k}_42/metadata.json").read_text()
        )
        for split, style in [("train", "-"), ("val", "--")]:
            ax.plot(
                history.epoch,
                history[split + "_macro_f1"],
                style,
                color="black",
                label=split,
            )
        ax.plot(
            testcurve.epoch,
            testcurve.macro_f1,
            ":",
            color=".5",
            label="test (retrospective)",
        )
        ax.axvline(
            metadata["best_epoch"], ls="-.", color=".7", label="validation choice"
        )
        ax.set_title(NAMES[k])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro-F1")
        ax.set_ylim(0.3, 1)
    axes[0, 0].legend(fontsize=9)
    fig.tight_layout()
    save("learning_curves")
    events = frame.iloc[ids].event_name.to_numpy()
    eventrows = []
    for event in sorted(set(events)):
        keep = events == event
        for k, p in probs.items():
            from sklearn.metrics import f1_score

            eventrows.append(
                {
                    "event": event,
                    "model": k,
                    "n": int(keep.sum()),
                    "macro_f1": f1_score(
                        y[keep], p[keep].argmax(1), average="macro", zero_division=0
                    ),
                }
            )
    eventtable = pd.DataFrame(eventrows)
    eventtable.to_csv(ROOT / "results/event_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for j, k in enumerate(["image", "concat_drop", "gated"]):
        v = eventtable[eventtable.model == k]
        axes[0].plot(
            np.arange(len(v)),
            v.macro_f1,
            marker=["o", "s", "x"][j],
            color=str(j * 0.3),
            ls=["-", "--", ":"][j],
            label=NAMES[k],
        )
    axes[0].set_xticks(
        np.arange(len(v)),
        [e.replace("_", " ") for e in v.event],
        rotation=35,
        ha="right",
        fontsize=10,
    )
    axes[0].set_ylabel("Test macro-F1")
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 1)
    for condition, style in [("clean", "-"), ("shuffled_text", "--")]:
        g = np.mean(
            [
                np.load(ROOT / f"results/predictions/gated_{s}.npz")[
                    "gate_" + condition
                ]
                for s in [42, 43, 44]
            ],
            0,
        )
        axes[1].hist(
            g,
            bins=25,
            histtype="step",
            ls=style,
            color="black",
            label=condition.replace("_", " "),
        )
    axes[1].set_xlabel("Mean text gate weight (seed ensemble)")
    axes[1].set_ylabel("Records")
    axes[1].legend(fontsize=10)
    fig.tight_layout()
    save("events_gates")
    # Schematic, not a measured attention map.
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.4, 3)
    ax.axis("off")

    def box(x, y, w, h, text):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.08",
                facecolor="white",
                edgecolor="black",
                linewidth=1,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    box(0.1, 1.8, 1.8, 0.7, "Text\nTF-IDF + SVD")
    box(0.1, 0.3, 1.8, 0.7, "Image\nfrozen ResNet18")
    box(2.5, 1.8, 2, 0.7, "128-D projection\n+ text availability mask")
    box(2.5, 0.3, 2, 0.7, "128-D projection\n+ image availability mask")
    box(5.1, 1.0, 2.3, 0.9, "Learned gate +\navailability normalisation")
    box(8, 1.0, 1.5, 0.9, "Fusion + masked\ninteractions")
    box(10.1, 1.0, 1.7, 0.9, "Classifier\ninformative / not")
    for start, end in [
        ((1.98, 2.15), (2.42, 2.15)),
        ((1.98, 0.65), (2.42, 0.65)),
        ((4.58, 2.15), (5.02, 1.65)),
        ((4.58, 0.65), (5.02, 1.25)),
        ((7.48, 1.45), (7.92, 1.45)),
        ((9.58, 1.45), (10.02, 1.45)),
    ]:
        ax.annotate(
            "", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "black"}
        )
    ax.text(
        6,
        -0.2,
        "Auxiliary unimodal heads and modality dropout are used during training.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout()
    save("architecture")
    print("Saved figures and summary tables", flush=True)
