import hashlib
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from portfolio_demo import study, verify_inputs
from scripts import verify_evidence


class DashboardTests(unittest.TestCase):
    def test_clean_metrics_use_three_seed_rows(self):
        data = study()
        self.assertEqual(data["test_size"], 3615)
        late = [row["f1"] for row in data["rows"]
                if row["model"] == "late_fusion" and row["condition"] == "clean"]
        self.assertEqual(len(late), 3)
        self.assertAlmostEqual(mean(late), 0.8168091109534309)
        self.assertAlmostEqual(data["test"]["permutation_p_two_sided"], 0.11478852114788521)

    def test_input_hash_must_match(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "weights"
            manifest = root / "manifest"
            artifact.write_bytes(b"expected")
            manifest.write_text(f"{hashlib.sha256(b'expected').hexdigest()}  weights\n")
            verify_inputs(root, manifest)
            artifact.write_bytes(b"different model")
            with self.assertRaisesRegex(ValueError, "differs from the published capstone"):
                verify_inputs(root, manifest)


class EvidenceTests(unittest.TestCase):
    def test_replay_submitted_accuracy_must_match_recorded_metrics(self):
        read_csv = verify_evidence.read_csv

        def altered_csv(name):
            rows = read_csv(name)
            if name == "full_reproduction_comparison.csv":
                rows[0]["accuracy_submitted"] = "0.1"
            return rows

        with patch.object(verify_evidence, "read_csv", side_effect=altered_csv):
            with self.assertRaisesRegex(AssertionError, "disagrees with metrics.csv"):
                verify_evidence.verify()
