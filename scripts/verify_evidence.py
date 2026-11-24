"""Check checked-in experimental evidence without raw images or retraining.

This is a consistency check of the archived aggregate measurements, not an
independent replication of model performance. It uses only Python stdlib.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def read_csv(name):
    with (ROOT / "results" / name).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def verify():
    protocol = read_json("protocol.json")
    audit = read_json("results/data_audit.json")
    stats = read_json("results/statistics.json")
    reprod = read_json("results/full_reproduction_validation.json")
    metrics = read_csv("metrics.csv")
    ensembles = read_csv("ensemble_metrics.csv")

    seeds = set(protocol["seeds"])
    assert seeds == {42, 43, 44}, f"Unexpected selected seeds: {seeds}"
    models = set(protocol["models"]) | {"late_fusion"}
    conditions = set(protocol["conditions"])
    assert conditions == {"clean", "no_text", "no_image", "shuffled_text"}
    assert audit["rows"] == 18082
    assert sum(sum(classes.values()) for classes in audit["splits"].values()) == audit["rows"]
    test_rows = sum(classes["test"] for classes in audit["splits"].values())
    assert test_rows == 3615

    rows_by_key = defaultdict(list)
    for item in metrics:
        key = (item["model"], item["condition"])
        assert item["model"] in models
        assert int(item["seed"]) in seeds
        assert item["condition"] in conditions
        for metric in ["accuracy", "macro_f1", "macro_precision", "macro_recall", "auroc"]:
            score = float(item[metric])
            assert math.isfinite(score) and 0 <= score <= 1, (key, metric, score)
        rows_by_key[key].append(item)
    assert set(rows_by_key) == {(model, condition) for model in models for condition in conditions}
    assert all(len(rows) == len(seeds) and {int(row["seed"]) for row in rows} == seeds
               for rows in rows_by_key.values())
    assert len(ensembles) == len(rows_by_key)
    assert {(row["model"], row["condition"]) for row in ensembles} == set(rows_by_key)

    ensemble_f1 = {(row["model"], row["condition"]): float(row["macro_f1"])
                   for row in ensembles}
    clean = {model: ensemble_f1[model, "clean"] for model in models}
    assert set(clean) == models
    assert max(clean, key=clean.get) == "late_fusion"
    assert clean["late_fusion"] > clean["gated"]

    # Paired clean comparison is prespecified in the submitted protocol.
    result = stats["clean"]
    expected_difference = clean["gated"] - clean["concat_drop"]
    assert math.isclose(result["difference_macro_f1"], expected_difference, abs_tol=1e-12)
    low, high = result["bootstrap_ci95"]
    assert low < 0 < high, "The primary confidence interval must not exclude zero"
    assert result["permutation_p_two_sided"] > 0.05
    assert set(stats) == conditions
    for condition, comparison in stats.items():
        difference = ensemble_f1["gated", condition] - ensemble_f1["concat_drop", condition]
        assert math.isclose(comparison["difference_macro_f1"], difference, abs_tol=1e-12), (
            f"Ensemble scores disagree with the paired difference for {condition}"
        )
        assert 0 < comparison["permutation_p_two_sided"] <= 1
        assert 1000 < comparison["groups"] <= test_rows
        assert comparison["repetitions"] == 10000
        lower, upper = comparison["bootstrap_ci95"]
        assert -1 <= lower <= upper <= 1

    secondary = sorted(
        conditions - {"clean"}, key=lambda condition: stats[condition]["permutation_p_two_sided"]
    )
    adjusted = 0.0
    for remaining, condition in zip(range(len(secondary), 0, -1), secondary, strict=True):
        comparison = stats[condition]
        adjusted = min(1.0, max(adjusted, remaining * comparison["permutation_p_two_sided"]))
        assert math.isclose(comparison["secondary_holm_p"], adjusted, abs_tol=1e-12), (
            f"Holm-adjusted p-value disagrees with the recorded p-values for {condition}"
        )

    assert reprod["status"] == "passed"
    assert reprod["features_and_models_rebuilt"] is True
    assert reprod["split_identical"] is True
    assert reprod["submitted_measurements_overwritten"] is False
    assert reprod["top_clean_mean_model_submitted"] == "late_fusion"
    assert reprod["top_clean_mean_model_reproduction"] == "late_fusion"

    comparison = read_csv("full_reproduction_comparison.csv")
    assert comparison, "Missing replay comparisons"
    key_fields = ("model", "seed", "condition")
    recorded = {tuple(row[field] for field in key_fields): row for row in metrics}
    replay = {tuple(row[field] for field in key_fields): row for row in comparison}
    assert len(replay) == len(comparison), "Duplicate rerun comparison rows"
    assert replay.keys() == recorded.keys(), "Rerun comparison coverage differs from metrics.csv"
    for key, row in replay.items():
        for metric in ("macro_f1", "accuracy"):
            submitted = float(row[f"{metric}_submitted"])
            reproduced = float(row[f"{metric}_reproduction"])
            assert math.isclose(submitted, float(recorded[key][metric]), rel_tol=0, abs_tol=1e-12), (
                f"Rerun comparison disagrees with metrics.csv for {key} ({metric})"
            )
            assert math.isfinite(reproduced) and 0 <= reproduced <= 1, (key, metric, reproduced)

    clean_replay = defaultdict(list)
    for row in comparison:
        if row["condition"] == "clean":
            clean_replay[row["model"]].append(row)
    maximum_difference = max(
        abs(mean(float(row["macro_f1_submitted"]) for row in rows) -
            mean(float(row["macro_f1_reproduction"]) for row in rows))
        for rows in clean_replay.values()
    )
    assert math.isclose(
        maximum_difference,
        reprod["maximum_clean_seed_mean_macro_f1_difference"],
        abs_tol=1e-12,
    )

    return {
        "status": "passed",
        "kind": "checked-in aggregate evidence consistency, NOT independent retraining",
        "test_rows": test_rows,
        "trained_seed_count": len(seeds),
        "model_families": len(protocol["models"]),
        "measured_model_conditions": len(rows_by_key),
        "clean_ensemble_macro_f1": clean,
        "primary_gated_vs_concat_difference": expected_difference,
        "primary_two_sided_p_value": stats["clean"]["permutation_p_two_sided"],
        "interval_excludes_zero": not (low <= 0 <= high),
        "secondary_holm_p_values": {condition: stats[condition]["secondary_holm_p"]
                                    for condition in secondary},
        "full_rerun_recorded": reprod["status"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
