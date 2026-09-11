import unittest
from unittest.mock import patch
import os
import tempfile
from pathlib import Path
from advanced_benchmarks import (
    is_safe_code,
    generate_filler_text,
    load_presets_config,
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



class TestPresetsConfigAndAlias(unittest.TestCase):
    def setUp(self):
        load_presets_config.cache_clear()
        map_repo_to_preset_alias.cache_clear()

    def tearDown(self):
        load_presets_config.cache_clear()
        map_repo_to_preset_alias.cache_clear()

    def test_load_presets_config_nonexistent_path(self):
        config = load_presets_config("/nonexistent/file/path/model_presets.ini")
        self.assertIsNone(config)

    def test_load_presets_config_default_nonexistent(self):
        with patch.dict(os.environ, {"PRESETS_FILE": "/nonexistent/default/model_presets.ini"}):
            load_presets_config.cache_clear()
            config = load_presets_config(None)
            self.assertIsNone(config)

    def test_load_presets_config_caching(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("[TestModel]\nalias = TestAlias\n", encoding="utf-8")

            c1 = load_presets_config(str(presets_file))
            c2 = load_presets_config(str(presets_file))
            self.assertIsNotNone(c1)
            self.assertIs(c1, c2)
            self.assertEqual(load_presets_config.cache_info().hits, 1)

    def test_load_presets_config_corrupt_ini(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "corrupt.ini"
            presets_file.write_text("corrupt header without closing bracket\nkey = val\n", encoding="utf-8")

            config = load_presets_config(str(presets_file))
            self.assertIsNone(config)

    def test_load_presets_config_alias_pointer(self):
        self.assertIs(_get_presets_config, load_presets_config)

    def test_map_repo_to_preset_alias_invalid_input(self):
        self.assertIsNone(map_repo_to_preset_alias(None))
        self.assertEqual(map_repo_to_preset_alias(""), "")
        self.assertEqual(map_repo_to_preset_alias(12345), 12345)

    def test_map_repo_to_preset_alias_nonexistent_file(self):
        res = map_repo_to_preset_alias("my-model", presets_file="/nonexistent/model_presets.ini")
        self.assertEqual(res, "my-model")

    def test_map_repo_to_preset_alias_matching_section(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("[Qwen3.6-27B]\nthreads = 8\n", encoding="utf-8")

            # Exact match case-insensitive
            res1 = map_repo_to_preset_alias("qwen3.6-27b", presets_file=str(presets_file))
            self.assertEqual(res1, "Qwen3.6-27B")

            res2 = map_repo_to_preset_alias("QWEN3.6-27B", presets_file=str(presets_file))
            self.assertEqual(res2, "Qwen3.6-27B")

    def test_map_repo_to_preset_alias_fallback_hf_repo_and_alias(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("""[*]
flash-attn = true

[Qwen3.6-27B-spec3]
hf-repo = unsloth/Qwen3.6-27B-spec3-GGUF
alias = qwen-spec-alias
""", encoding="utf-8")

            # hf-repo match
            res_repo = map_repo_to_preset_alias("unsloth/qwen3.6-27b-spec3-gguf", presets_file=str(presets_file))
            self.assertEqual(res_repo, "Qwen3.6-27B-spec3")

            # alias match
            res_alias = map_repo_to_preset_alias("QWEN-SPEC-ALIAS", presets_file=str(presets_file))
            self.assertEqual(res_alias, "Qwen3.6-27B-spec3")

    def test_map_repo_to_preset_alias_fallback_no_match(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("[Qwen3.6-27B]\nthreads = 8\n", encoding="utf-8")

            res = map_repo_to_preset_alias("completely-unknown-repo", presets_file=str(presets_file))
            self.assertEqual(res, "completely-unknown-repo")

    def test_map_repo_to_preset_alias_caching(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("[CachedModel]\nalias = CM\n", encoding="utf-8")

            res1 = map_repo_to_preset_alias("CachedModel", presets_file=str(presets_file))
            res2 = map_repo_to_preset_alias("CachedModel", presets_file=str(presets_file))
            self.assertEqual(res1, "CachedModel")
            self.assertEqual(res2, "CachedModel")
            self.assertEqual(map_repo_to_preset_alias.cache_info().hits, 1)

    def test_map_repo_to_preset_alias_corrupt_ini_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "corrupt.ini"
            presets_file.write_text("[invalid ini", encoding="utf-8")

            res = map_repo_to_preset_alias("fallback-model", presets_file=str(presets_file))
            self.assertEqual(res, "fallback-model")

    def test_get_preset_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("""[*]
flash-attn = true
parallel = 2

[MyProfile]
parallel = 4
n-gpu-layers = 33
custom-param = hello
""", encoding="utf-8")

            meta = get_preset_metadata("MyProfile", presets_file=str(presets_file))
            self.assertEqual(meta["flash_attn"], "true")
            self.assertEqual(meta["parallel"], "4")
            self.assertEqual(meta["n_gpu_layers"], "33")
            self.assertEqual(meta["custom_param"], "hello")

            # Nonexistent section still gets globals and defaults
            meta_default = get_preset_metadata("UnknownProfile", presets_file=str(presets_file))
            self.assertEqual(meta_default["flash_attn"], "true")
            self.assertEqual(meta_default["parallel"], "2")
            self.assertEqual(meta_default["n_gpu_layers"], "99")

    def test_get_preset_metadata_nonexistent_file(self):
        meta = get_preset_metadata("AnyProfile", presets_file="/nonexistent/model_presets.ini")
        self.assertEqual(meta["parallel"], "1")
        self.assertEqual(meta["n_gpu_layers"], "99")
        self.assertEqual(meta["spec_type"], "None")

if __name__ == "__main__":
    unittest.main()
