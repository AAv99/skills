import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "style_discovery.py"
spec = importlib.util.spec_from_file_location("style_discovery", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class StyleDiscoveryTests(unittest.TestCase):
    def test_log_ratio_direction(self):
        self.assertGreater(module.log_ratio(20, 100, 5, 100), 0)
        self.assertLess(module.log_ratio(5, 100, 20, 100), 0)

    def test_g2_equal_rates_is_zero(self):
        self.assertAlmostEqual(module.g2(10, 100, 20, 200), 0.0, places=10)

    def test_g2_detects_large_difference(self):
        self.assertGreater(module.g2(40, 100, 5, 100), 20)

    def test_lexical_extraction_keeps_function_words_and_char_ngrams(self):
        result = module.extract("Dit is een korte zin, maar hij is helder.", None, include_syntax=False)
        self.assertGreater(result.counts["function_word"]["dit"], 0)
        self.assertGreater(result.counts["function_word"]["maar"], 0)
        self.assertIn("Dit", result.counts["char_3gram"])
        self.assertEqual(result.metrics, {})

    def test_load_unpaired_records_preserves_text_and_skips_empty_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "texts.json"
            path.write_text(
                json.dumps(
                    [
                        {"id": "one", "text": "Casing, punctuation!"},
                        {"id": "empty", "text": ""},
                        {"id": "missing", "other": "not text"},
                    ]
                ),
                encoding="utf-8",
            )
            records = module.load_texts(path, "text", "id")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].text_id, "one")
            self.assertEqual(records[0].text, "Casing, punctuation!")

    def test_descriptive_profile_has_no_reference_statistics(self):
        records = [
            module.TextRecord("one", "Dit is een korte zin."),
            module.TextRecord("two", "Dit is een tweede zin."),
        ]
        extracted = [module.extract(record.text, None, include_syntax=False) for record in records]
        payload = module.build_descriptive_profile(records, extracted, "", min_count=1)
        self.assertEqual(payload["mode"], "descriptive")
        self.assertEqual(payload["record_count"], 2)
        self.assertGreater(payload["corpus_stats"]["tokens"], 0)
        self.assertTrue(payload["features"])
        self.assertFalse(any("log_ratio" in row or "g2" in row for row in payload["features"]))

    def test_surface_concordance_keeps_case_and_character_spacing(self):
        snippet = module.surface_concordance("Begin, X en Y.", " en", "char_3gram")
        self.assertIsNotNone(snippet)
        self.assertIn("X en Y", snippet)

    def test_descriptive_cli_writes_report_without_reference_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "texts.json"
            corpus.write_text(
                json.dumps(
                    [
                        {"id": "1", "text": "Maar dit is helder."},
                        {"id": "2", "text": "Dit is ook helder."},
                    ]
                ),
                encoding="utf-8",
            )
            out = root / "out"
            run = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    str(corpus),
                    "--mode",
                    "descriptive",
                    "--no-syntax",
                    "--min-count",
                    "1",
                    "--out-dir",
                    str(out),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            payload = json.loads((out / "style-discovery.json").read_text(encoding="utf-8"))
            report = (out / "style-discovery.md").read_text(encoding="utf-8")
            self.assertEqual(payload["mode"], "descriptive")
            self.assertNotIn("log_ratio", json.dumps(payload))
            self.assertIn("No reference corpus", report)

    def test_load_pairs_skips_incomplete_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"pair_id": "1", "draft": "A", "final": "B"}),
                        json.dumps({"pair_id": "2", "draft": "", "final": "B"}),
                    ]
                ),
                encoding="utf-8",
            )
            pairs = module.load_pairs(path, "draft", "final", "pair_id")
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].pair_id, "1")

    def test_pair_evidence_rewards_repeated_direction(self):
        pairs = [module.Pair("1", "", ""), module.Pair("2", "", "")]
        ea = [
            module.Extracted({"function_word": module.Counter({"maar": 4, "de": 6})}, {}),
            module.Extracted({"function_word": module.Counter({"maar": 3, "de": 7})}, {}),
        ]
        eb = [
            module.Extracted({"function_word": module.Counter({"maar": 1, "de": 9})}, {}),
            module.Extracted({"function_word": module.Counter({"maar": 1, "de": 9})}, {}),
        ]
        dispersion, consistency, observed, informative = module.pair_evidence(
            "maar", "function_word", pairs, ea, eb, 1
        )
        self.assertEqual(dispersion, 1.0)
        self.assertEqual(consistency, 1.0)
        self.assertEqual(observed, 2)
        self.assertEqual(informative, 2)

    def test_cli_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "pairs.jsonl"
            corpus.write_text(
                "\n".join(
                    [
                        json.dumps({"pair_id": "1", "draft": "Maar dit is echter wel zo.", "final": "Dit is zo."}),
                        json.dumps({"pair_id": "2", "draft": "Maar wij doen dit ook.", "final": "Wij doen dit."}),
                        json.dumps({"pair_id": "3", "draft": "Maar dat is ook helder.", "final": "Dat is helder."}),
                    ]
                ),
                encoding="utf-8",
            )
            out = root / "out"
            run = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    str(corpus),
                    "--no-syntax",
                    "--min-count",
                    "1",
                    "--out-dir",
                    str(out),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue((out / "style-discovery.json").exists())
            self.assertTrue((out / "style-discovery.md").exists())
            payload = json.loads((out / "style-discovery.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["pair_count"], 3)
            self.assertTrue(payload["features"])


if __name__ == "__main__":
    unittest.main()
