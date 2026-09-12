import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from advanced_benchmarks import (
    is_safe_code,
    generate_filler_text,
    _get_presets_config,
    map_repo_to_preset_alias,
    get_preset_metadata,
)

class TestIsSafeCode(unittest.TestCase):
    def test_valid_code(self):
        code = "print('Hello World')\nx = 1 + 1"
        safe, msg = is_safe_code(code)
        self.assertTrue(safe)
        self.assertIsNone(msg)

    def test_syntax_error(self):
        code = "print('Hello World'"
        safe, msg = is_safe_code(code)
        self.assertFalse(safe)
        self.assertIn("Syntax error", msg)

    def test_forbidden_code_patterns(self):
        cases = [
            ("import subprocess", "import subprocess", "Forbidden"),
            ("import subprocess.Popen", "import sub-module", "Forbidden"),
            ("from subprocess import Popen", "from forbidden module", "Forbidden"),
            ("from os import system", "from forbidden name", "Forbidden"),
            ("import os\nos.system('ls')", "forbidden attribute usage", "Forbidden"),
            ("eval('1 + 1')", "forbidden identifier usage", "Forbidden"),
        ]
        for code, label, expected_keyword in cases:
            with self.subTest(case=label):
                safe, msg = is_safe_code(code)
                self.assertFalse(safe)
                self.assertIsNotNone(msg)
                self.assertIn(expected_keyword, msg)

class TestGenerateFillerText(unittest.TestCase):
    def test_zero_target(self):
        res = generate_filler_text(0)
        self.assertEqual(res, [])

    def test_negative_target(self):
        res = generate_filler_text(-10)
        self.assertEqual(res, [])

    def test_positive_target_size(self):
        # target_tokens=10 => target_chars=45
        res = generate_filler_text(10)
        self.assertTrue(len(res) > 0)
        self.assertTrue(all(isinstance(p, str) for p in res))
        total_chars = sum(len(p) + 1 for p in res)
        self.assertGreaterEqual(total_chars, 45)

    @patch('advanced_benchmarks.random.choice')
    def test_paragraph_structure(self, mock_choice):
        mock_choice.return_value = "A sentence."
        # One paragraph of 5 sentences is "A sentence. A sentence. A sentence. A sentence. A sentence."
        # len = 5 * 11 + 4 = 59.
        # Target tokens = 10 => 45 chars.
        # This should generate exactly 1 paragraph.
        res = generate_filler_text(10)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], "A sentence. A sentence. A sentence. A sentence. A sentence.")

class TestPresetsCachingAndMapping(unittest.TestCase):
    def setUp(self):
        _get_presets_config.cache_clear()

    def tearDown(self):
        _get_presets_config.cache_clear()

    def test_presets_config_caching(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("[*]\nparallel = 2\n\n[Qwen3.6-27B]\nalias = qwen27b\nhf-repo = unsloth/qwen3.6-27b\n", encoding="utf-8")

            cfg1 = _get_presets_config(str(presets_file))
            self.assertIsNotNone(cfg1)
            self.assertIn("Qwen3.6-27B", cfg1.sections())

            # Modify file content on disk without clearing cache
            presets_file.write_text("[*]\nparallel = 4\n\n[NewSection]\nalias = new\n", encoding="utf-8")

            # Subsequent call should return cached object
            cfg2 = _get_presets_config(str(presets_file))
            self.assertIs(cfg1, cfg2)
            self.assertNotIn("NewSection", cfg2.sections())

            # After cache clear, new content should be loaded
            _get_presets_config.cache_clear()
            cfg3 = _get_presets_config(str(presets_file))
            self.assertIsNot(cfg1, cfg3)
            self.assertIn("NewSection", cfg3.sections())

    def test_presets_config_nonexistent_and_malformed_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            non_existent = str(Path(tmp_dir) / "missing.ini")
            cfg_missing = _get_presets_config(non_existent)
            self.assertIsNone(cfg_missing)

            # Check mapping and metadata fallback when file missing
            self.assertEqual(map_repo_to_preset_alias("Qwen3.6-27B", presets_file=non_existent), "Qwen3.6-27B")
            meta = get_preset_metadata("Qwen3.6-27B", presets_file=non_existent)
            self.assertEqual(meta["parallel"], "1")

            # Malformed file
            malformed_file = Path(tmp_dir) / "malformed.ini"
            malformed_file.write_text("invalid [ini content without closing bracket", encoding="utf-8")

            _get_presets_config.cache_clear()
            cfg_malformed = _get_presets_config(str(malformed_file))
            self.assertIsNone(cfg_malformed)
            self.assertEqual(map_repo_to_preset_alias("Qwen3.6-27B", presets_file=str(malformed_file)), "Qwen3.6-27B")

    def test_map_repo_to_preset_alias_and_metadata_with_presets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text(
                "[*]\nflash-attn = true\nparallel = 2\n\n"
                "[Qwen3.6-27B]\nhf-repo = unsloth/Qwen3.6-27B-GGUF\nalias = qwen27b\nspec-type = ngram\n",
                encoding="utf-8"
            )

            # Match by section name
            self.assertEqual(map_repo_to_preset_alias("qwen3.6-27b", presets_file=str(presets_file)), "Qwen3.6-27B")
            # Match by hf-repo
            self.assertEqual(map_repo_to_preset_alias("unsloth/Qwen3.6-27B-GGUF", presets_file=str(presets_file)), "Qwen3.6-27B")
            # Match by alias
            self.assertEqual(map_repo_to_preset_alias("qwen27b", presets_file=str(presets_file)), "Qwen3.6-27B")
            # Fallback for unknown
            self.assertEqual(map_repo_to_preset_alias("UnknownModel", presets_file=str(presets_file)), "UnknownModel")

            # Metadata parsing
            meta = get_preset_metadata("Qwen3.6-27B", presets_file=str(presets_file))
            self.assertEqual(meta["flash_attn"], "true")
            self.assertEqual(meta["parallel"], "2")
            self.assertEqual(meta["spec_type"], "ngram")


if __name__ == "__main__":
    unittest.main()
