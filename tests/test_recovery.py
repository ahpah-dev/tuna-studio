"""Damaged optional local metadata must not take the app offline."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import APIError, Studio, atomic_json


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.studio = Studio(self.root, probe=False)

    def tearDown(self):
        self.studio.close()
        self.temp.cleanup()

    def test_settings_recovers_nonobjects_and_invalid_fields(self):
        path = self.studio.data_dir / "settings.json"
        for saved in (None, [], "broken", {"offline": "false", "default_model": [], "theme": None}):
            with self.subTest(saved=saved):
                atomic_json(path, saved)
                self.assertEqual(self.studio.settings(), {"offline": False, "default_model": "", "theme": "light"})
                self.assertFalse(self.studio.get_status()["offline"])
        atomic_json(path, {"offline": True, "theme": "dark", "default_model": 123})
        self.assertEqual(self.studio.settings(), {"offline": True, "default_model": "", "theme": "dark"})
        self.assertEqual(self.studio.settings({"default_model": "local/model"})["default_model"], "local/model")
        self.assertTrue(self.studio.settings()["offline"])

    def test_corrupt_dataset_is_skipped_without_hiding_healthy_datasets(self):
        valid = self.studio.add_dataset({"name": "good.jsonl", "content": '{"prompt":"q","completion":"a"}'})
        identifier = "a" * 32
        metadata = self.studio.datasets_dir / (identifier + ".meta.json")
        for saved in (None, [], {}, {**valid, "id": identifier}, {**valid, "id": identifier, "rows": "1"}):
            with self.subTest(saved=saved):
                atomic_json(metadata, saved)
                self.assertEqual([item["id"] for item in self.studio.list_datasets()], [valid["id"]])
                with self.assertRaises(APIError) as error:
                    self.studio.dataset(identifier)
                self.assertEqual(error.exception.status, 404)
        # An existing metadata file alone is not a usable training dataset.
        (self.studio.datasets_dir / (valid["id"] + ".jsonl")).unlink()
        self.assertEqual(self.studio.list_datasets(), [])

    def test_run_recovery_handles_optional_damage_and_rejects_id_mismatch(self):
        identifier = "b" * 32
        run = {"id": identifier, "status": "completed", "created_at": "2026-01-01", "config": {},
               "gguf": None, "metrics": None, "logs": [None, {"message": "kept"}]}
        atomic_json(self.studio.runs_dir / identifier / "state.json", run)
        atomic_json(self.studio.runs_dir / ("c" * 32) / "state.json", run)
        atomic_json(self.studio.runs_dir / ("d" * 32) / "state.json", [])
        atomic_json(self.studio.runs_dir / ("e" * 32) / "state.json", {"id": "e" * 32})
        recovered = Studio(self.root, probe=False)
        try:
            self.assertEqual(len(recovered.list_runs()), 1)
            value = recovered.get_run(identifier)
            self.assertEqual(value["metrics"], [])
            self.assertEqual(value["logs"], [{"message": "kept"}])
            self.assertEqual(value["gguf"]["result"], {})
            self.assertIsNone(recovered.get_status()["active_run_id"])
        finally:
            recovered.close()


if __name__ == "__main__":
    unittest.main()
