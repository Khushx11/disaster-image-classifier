"""Train on training data; select checkpoints exclusively on validation macro-F1."""

import json, time, random, copy, platform, hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from .models import build_model, training_availability
from .data import ROOT
from .features import validate_feature_rows


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def score(model, text, image, y):
    model.eval()
    with torch.inference_mode():
        logits = model(text, image)[0]
        loss = torch.nn.functional.cross_entropy(logits, y).item()
        pred = logits.argmax(1).cpu().numpy()
    return {
        "loss": loss,
        "macro_f1": f1_score(y.cpu().numpy(), pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y.cpu().numpy(), pred),
    }


def run(only=None):
    torch.set_num_threads(4)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    cfg = protocol["training"]
    features = np.load(ROOT / "work/features/features.npz", allow_pickle=True)
    validate_feature_rows(features, ROOT)
    # Small cached heads run efficiently on CPU and avoid MPS nondeterminism.
    text = torch.tensor(features["text"])
    image = torch.tensor(features["image"])
    y = torch.tensor(features["y"], dtype=torch.long)
    tr = np.where(features["split"] == "train")[0]
    va = np.where(features["split"] == "val")[0]
    weights = torch.tensor(
        len(tr) / (2 * np.bincount(features["y"][tr], minlength=2)), dtype=torch.float32
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
    output = ROOT / "results/models"
    output.mkdir(parents=True, exist_ok=True)
    epochs_path = ROOT / "work/epochs"
    epochs_path.mkdir(parents=True, exist_ok=True)
    for kind in protocol["models"]:
        if only and kind not in only:
            continue
        for seed in protocol["seeds"]:
            name = f"{kind}_{seed}"
            directory = output / name
            if (directory / "metadata.json").exists():
                continue
            directory.mkdir(exist_ok=True)
            seed_all(seed)
            model = build_model(
                kind, protocol, text_dim=text.shape[1], image_dim=image.shape[1]
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
            )
            history = []
            states = []
            best = -1
            best_epoch = 0
            start = time.perf_counter()
            for epoch in range(cfg["epochs"]):
                model.train()
                order = torch.tensor(tr)[torch.randperm(len(tr))]
                loss_sum = 0
                for ids in order.split(cfg["batch_size"]):
                    available = training_availability(len(ids), kind, cfg)
                    logits, aux, _ = model(text[ids], image[ids], available)
                    loss = criterion(logits, y[ids]).mean()
                    if aux is not None and kind != "no_aux":
                        losses = torch.stack(
                            [criterion(aux[0], y[ids]), criterion(aux[1], y[ids])], 1
                        )
                        loss = (
                            loss
                            + cfg["auxiliary_weight"]
                            * (losses * available).sum()
                            / available.sum()
                        )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item() * len(ids)
                training = score(model, text[tr], image[tr], y[tr])
                validation = score(model, text[va], image[va], y[va])
                history.append(
                    {
                        "epoch": epoch + 1,
                        "objective": loss_sum / len(tr),
                        **{"train_" + k: v for k, v in training.items()},
                        **{"val_" + k: v for k, v in validation.items()},
                    }
                )
                state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                states.append(state)
                if validation["macro_f1"] > best + 1e-12:
                    best = validation["macro_f1"]
                    best_epoch = epoch + 1
                    torch.save(state, directory / "best.pt")
                if epoch + 1 >= 20 and epoch + 1 - best_epoch >= cfg["patience"]:
                    break
            torch.save(states, epochs_path / f"{name}.pt")
            (directory / "history.json").write_text(json.dumps(history, indent=2))
            metadata = {
                "kind": kind,
                "seed": seed,
                "best_epoch": best_epoch,
                "validation_macro_f1": best,
                "epochs": len(history),
                "seconds": time.perf_counter() - start,
                "parameters": sum(p.numel() for p in model.parameters()),
                "device": "cpu",
                "torch": torch.__version__,
                "python": platform.python_version(),
                "test_used_for_selection": False,
            }
            (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
            print(name, metadata, flush=True)
    assert all(
        (output / f"{k}_{s}/metadata.json").exists()
        for k in protocol["models"]
        for s in protocol["seeds"]
    )
    (output / "selection_locked.json").write_text(
        json.dumps(
            {
                "protocol": protocol,
                "checkpoint_sha256": {
                    f"{k}_{s}": hashlib.sha256(
                        (output / f"{k}_{s}/best.pt").read_bytes()
                    ).hexdigest()
                    for k in protocol["models"]
                    for s in protocol["seeds"]
                },
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "test_used_for_selection": False,
            },
            indent=2,
        )
    )
