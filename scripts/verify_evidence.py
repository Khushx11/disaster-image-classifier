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
        assert item["model"] in protocol["models"]
        assert int(item["seed"]) in seeds
        assert item["condition"] in conditions
        for metric in ["accuracy", "macro_f1", "macro_precision", "macro_recall", "auroc"]:
            score = float(item[metric])
            assert math.isfinite(score) and 0 <= score <= 1, (key, metric, score)
        rows_by_key[key].append(item)
    assert len(rows_by_key) == len(protocol["models"]) * len(conditions)
    assert all(len(rows) == len(seeds) for rows in rows_by_key.values())

    clean = {item["model"]: float(item["macro_f1"])
             for item in ensembles if item["condition"] == "clean"}
    assert set(clean) == models
    assert max(clean, key=clean.get) == "late_fusion"
    assert clean["late_fusion"] > clean["gated"]

    # Paired clean comparison is prespecified in the submitted protocol.
    result = stats["clean"]
    expected_difference = clean["gated"] - clean["concat_drop"]
    assert math.isclose(result["difference_macro_f1"], expected_difference, abs_tol=1e-12)
    low, high = result["bootstrap_ci95"]
    assert low < 0 < high, "The primary confidence interval must not exclude zero"
    assert 0 < result["permutation_p_two_sided"] < 1
    assert result["permutation_p_two_sided"] > 0.05
    for condition in conditions:
        result = stats[condition]
        assert result["groups"] > 1000
        assert result["repetitions"] == 10000
        assert len(result["bootstrap_ci95"]) == 2
        assert result["bootstrap_ci95"][0] <= result["bootstrap_ci95"][1]

    assert reprod["status"] == "passed"
    assert reprod["features_and_models_rebuilt"] is True
    assert reprod["split_identical"] is True
    assert reprod["submitted_measurements_overwritten"] is False
    assert reprod["top_clean_mean_model_submitted"] == "late_fusion"
    assert reprod["top_clean_mean_model_reproduction"] == "late_fusion"

    comparison = read_csv("full_reproduction_comparison.csv")
    assert comparison, "Missing replay comparisons"
    assert {(r["model"], r["seed"], r["condition"]) for r in comparison} == {
        (r["model"], r["seed"], r["condition"]) for r in metrics
    }
    maximum_difference = max(abs(float(row["macro_f1_submitted"]) -
                                 float(row["macro_f1_reproduction"]))
                             for row in comparison)
    assert math.isclose(
        maximum_difference,
        reprod["maximum_clean_seed_mean_macro_f1_difference"],
        abs_tol=0.02,
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
        "full_rerun_recorded": reprod["status"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
