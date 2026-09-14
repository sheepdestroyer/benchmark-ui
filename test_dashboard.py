import configparser
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


class DictWithDefault(dict):
    def __getattr__(self, name):
        return self.get(name, None)

    def __setattr__(self, name, value):
        self[name] = value


mock_st = MagicMock()
mock_st.session_state = DictWithDefault(
    {
        "new_endpoint": "http://localhost:8080",
        "new_model": "test-model",
        "new_api_key": "sk-1234",
        "endpoints": [],
        "suite_results": None,
        "matrix_results": None,
        "kld_results": None,
    }
)


def mock_tabs(*args, **kwargs):
    titles = args[0] if len(args) > 0 else []
    mocks = []
    for _ in titles:
        m = MagicMock()
        m.__enter__ = MagicMock(return_value=m)
        m.__exit__ = MagicMock(return_value=None)
        mocks.append(m)
    return mocks


mock_st.tabs = mock_tabs


def mock_columns(*args, **kwargs):
    spec = args[0] if len(args) > 0 else 1
    num = spec if isinstance(spec, int) else len(spec)
    mocks = []
    for _ in range(num):
        m = MagicMock()
        m.__enter__ = MagicMock(return_value=m)
        m.__exit__ = MagicMock(return_value=None)
        mocks.append(m)
    return mocks


mock_st.columns = mock_columns

mock_sidebar = MagicMock()
mock_sidebar.__enter__ = MagicMock(return_value=mock_sidebar)
mock_sidebar.__exit__ = MagicMock(return_value=None)
mock_st.sidebar = mock_sidebar


def mock_text_input(label, *args, **kwargs):
    if "Endpoint URL" in label:
        return "http://localhost:8080"
    if "Model Name" in label:
        return "test-model"
    if "API Key" in label:
        return "sk-1234"
    if "Corpus Name" in label:
        return "kld_corpus.txt"
    if "GGUF Path" in label:
        return ""
    return "text"


mock_st.text_input = mock_text_input


def mock_selectbox(label, *args, **kwargs):
    return "test-model"


mock_st.selectbox = mock_selectbox


def mock_form(*args, **kwargs):
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__ = MagicMock(return_value=None)
    return m


mock_st.form = mock_form

mock_expander = MagicMock()
mock_expander.__enter__ = MagicMock(return_value=mock_expander)
mock_expander.__exit__ = MagicMock(return_value=None)
mock_st.expander = MagicMock(return_value=mock_expander)

mock_st.button.return_value = False


class StreamlitMock(MagicMock):
    def columns(self, num, *args, **kwargs):
        if isinstance(num, int):
            return [MagicMock() for _ in range(num)]
        if isinstance(num, (list, tuple)):
            return [MagicMock() for _ in num]
        return [MagicMock(), MagicMock()]

    def tabs(self, tabs, *args, **kwargs):
        return [MagicMock() for _ in tabs]

    def text_input(self, label, value="", *args, **kwargs):
        return value

    def number_input(self, label, *args, **kwargs):
        return kwargs.get("value", 5000)

    def selectbox(self, label, options, *args, **kwargs):
        if options:
            return options[0]
        return "mock_model"

    def button(self, *args, **kwargs):
        return False


mock_st = StreamlitMock()

modules_patcher = patch.dict(
    sys.modules,
    {
        "streamlit": mock_st,
        "pandas": MagicMock(),
        "plotly": MagicMock(),
        "plotly.express": MagicMock(),
        "plotly.graph_objects": MagicMock(),
    },
)
modules_patcher.start()
import dashboard

modules_patcher.stop()


class TestValidateEndpointUrl(unittest.TestCase):
    def test_valid_urls(self):
        self.assertEqual(
            dashboard.validate_endpoint_url("http://google.com"), "http://google.com"
        )
        self.assertEqual(
            dashboard.validate_endpoint_url("https://api.github.com/v1"),
            "https://api.github.com/v1",
        )
        self.assertEqual(
            dashboard.validate_endpoint_url("http://8.8.8.8"), "http://8.8.8.8"
        )

    def test_localhost_exceptions(self):
        self.assertEqual(
            dashboard.validate_endpoint_url("http://localhost"), "http://localhost"
        )
        self.assertEqual(
            dashboard.validate_endpoint_url("http://localhost:8080"),
            "http://localhost:8080",
        )
        self.assertEqual(
            dashboard.validate_endpoint_url("http://127.0.0.1"), "http://127.0.0.1"
        )
        self.assertEqual(
            dashboard.validate_endpoint_url("http://127.0.0.1:5000"),
            "http://127.0.0.1:5000",
        )

    def test_empty_url(self):
        with self.assertRaisesRegex(ValueError, r"cannot be empty"):
            dashboard.validate_endpoint_url("")
        with self.assertRaisesRegex(ValueError, r"cannot be empty"):
            dashboard.validate_endpoint_url(None)

    def test_invalid_scheme(self):
        with self.assertRaisesRegex(ValueError, r"Invalid URL scheme"):
            dashboard.validate_endpoint_url("ftp://server.com")
        with self.assertRaisesRegex(ValueError, r"Invalid URL scheme"):
            dashboard.validate_endpoint_url("ws://localhost")

    def test_missing_hostname(self):
        with self.assertRaisesRegex(ValueError, r"missing hostname"):
            dashboard.validate_endpoint_url("http://")
        with self.assertRaisesRegex(ValueError, r"missing hostname"):
            dashboard.validate_endpoint_url("http:/path/only")

    def test_forbidden_ips(self):
        forbidden_ips = [
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "224.0.0.1",
            "240.0.0.1",
        ]
        forbidden_ipv6 = ["[::1]", "[fe80::1]", "[fd00::1]"]

        for ip in forbidden_ips:
            with self.subTest(ip=ip):
                with self.assertRaisesRegex(ValueError, r"Forbidden IP"):
                    dashboard.validate_endpoint_url(f"http://{ip}")

        for ip in forbidden_ipv6:
            with self.subTest(ip=ip):
                with self.assertRaisesRegex(ValueError, r"Forbidden IP"):
                    dashboard.validate_endpoint_url(f"http://{ip}")

        with self.assertRaisesRegex(ValueError, r"Forbidden IP"):
            dashboard.validate_endpoint_url("http://127.0.0.2")


class TestDashboard(unittest.TestCase):
    def test_validate_model_name_valid(self):
        valid_names = [
            "llama2",
            "llama-2-7b",
            "meta-llama/Llama-2-7b-chat-hf",
            "model_v1.0",
            "model:latest",
            "my.model.name",
            "12345",
        ]
        for name in valid_names:
            with self.subTest(name=name):
                self.assertEqual(dashboard.validate_model_name(name), name)

    def test_validate_model_name_invalid(self):
        invalid_names = [
            "",
            None,
            "model name with spaces",
            "model@name",
            "model!name",
            "model#name",
            "model$name",
            "model%name",
            "model^name",
            "model&name",
            "model*name",
            "model(name)",
            "model+",
            "model=",
            "model<",
            "model>",
            "model?",
            "model`",
            "model~",
            "model{",
            "model}",
            "model[",
            "model]",
        ]
        for name in invalid_names:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    dashboard.validate_model_name(name)


class TestPresetConfig(unittest.TestCase):
    def setUp(self):
        dashboard._get_presets_config.cache_clear()

    def tearDown(self):
        dashboard._get_presets_config.cache_clear()

    @patch("os.path.exists", return_value=False)
    def test_get_presets_config_missing_file(self, mock_exists):
        config = dashboard._get_presets_config()
        self.assertIsNone(config)

    @patch("os.path.exists", return_value=True)
    def test_get_presets_config_syntax_error(self, mock_exists):
        with patch(
            "configparser.ConfigParser.read",
            side_effect=configparser.ParsingError("Invalid INI"),
        ):
            config = dashboard._get_presets_config()
            self.assertIsNone(config)

    @patch("os.path.exists", return_value=True)
    def test_get_presets_config_caching(self, mock_exists):
        sample_ini = """
[*]
flash-attn = true

[my-model]
alias = MyModel
hf-repo = org/my-model
parallel = 2
"""
        with patch("configparser.ConfigParser.read") as mock_read:

            def fake_read(filenames, encoding=None):
                # simulate successful read by populating sections
                return [filenames]

            mock_read.side_effect = fake_read

            c1 = dashboard._get_presets_config()
            c2 = dashboard._get_presets_config()
            self.assertIs(c1, c2)
            self.assertEqual(mock_read.call_count, 1)

    def test_get_presets_config_env_var(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_ini = Path(tmp_dir) / "custom_presets.ini"
            custom_ini.write_text("[EnvProfile]\nparallel = 5\n", encoding="utf-8")

            with patch.dict(os.environ, {"PRESETS_FILE": str(custom_ini)}):
                dashboard._get_presets_config.cache_clear()
                cfg = dashboard._get_presets_config()
                self.assertIsNotNone(cfg)
                self.assertIn("EnvProfile", cfg.sections())

    def test_get_presets_config_fallback_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fallback_ini = Path(tmp_dir) / "model_presets.ini"
            fallback_ini.write_text(
                "[FallbackProfile]\nthreads = 12\n", encoding="utf-8"
            )

            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(
                    dashboard, "resolve_presets_path", return_value=str(fallback_ini)
                ),
            ):
                dashboard._get_presets_config.cache_clear()
                cfg = dashboard._get_presets_config()
                self.assertIsNotNone(cfg)
                self.assertIn("FallbackProfile", cfg.sections())

    def test_dashboard_resolve_presets_path(self):
        # 1. Explicit path
        self.assertEqual(
            dashboard.resolve_presets_path("/explicit/path.ini"), "/explicit/path.ini"
        )
        self.assertEqual(
            dashboard.resolve_presets_path(Path("/explicit/path2.ini")),
            "/explicit/path2.ini",
        )

        # 2. PRESETS_FILE environment variable
        with patch.dict(os.environ, {"PRESETS_FILE": "/env/path.ini"}):
            self.assertEqual(dashboard.resolve_presets_path(None), "/env/path.ini")

        # 3. Default fallback logic when PRESETS_FILE is not set
        primary = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "../llama.cpp/profiles/model_presets.ini"
            )
        )
        fallback = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../llama.cpp/model_presets.ini")
        )

        # Primary exists
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("os.path.exists", side_effect=lambda p: str(p) == primary),
        ):
            self.assertEqual(dashboard.resolve_presets_path(None), primary)

        # Primary does not exist, fallback exists
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("os.path.exists", side_effect=lambda p: str(p) == fallback),
        ):
            self.assertEqual(dashboard.resolve_presets_path(None), fallback)

        # Neither exists -> returns primary
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("os.path.exists", return_value=False),
        ):
            self.assertEqual(dashboard.resolve_presets_path(None), primary)

    def test_map_repo_to_preset_alias_unsloth_prefix_normalization(self):
        cp = configparser.ConfigParser()
        cp.read_string("""
[qwen-profile]
hf-repo = unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_XL
alias = unsloth/locallama-qwen, local-qwen

[plain-profile]
hf-repo = PlainModel-GGUF:latest
alias = plain-alias

[unsloth/SectionProfile]
hf-repo = org/foo
""")
        with patch.object(dashboard, "_get_presets_config", return_value=cp):
            # 1. Query has NO prefix, hf-repo HAS prefix
            self.assertEqual(
                dashboard.map_repo_to_preset_alias("Qwen3.8-27B-GGUF:UD-Q5_K_XL"),
                "qwen-profile",
            )
            # 2. Query has prefix, hf-repo has prefix
            self.assertEqual(
                dashboard.map_repo_to_preset_alias(
                    "unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_XL"
                ),
                "qwen-profile",
            )
            # 3. Query has prefix, hf-repo has NO prefix
            self.assertEqual(
                dashboard.map_repo_to_preset_alias("unsloth/PlainModel-GGUF:latest"),
                "plain-profile",
            )
            # 4. Query has NO prefix, section HAS prefix
            self.assertEqual(
                dashboard.map_repo_to_preset_alias("SectionProfile"),
                "unsloth/SectionProfile",
            )
            # 5. Query has prefix, section has NO prefix
            self.assertEqual(
                dashboard.map_repo_to_preset_alias("unsloth/plain-profile"),
                "plain-profile",
            )
            # 6. Alias normalization: query has no prefix, alias has prefix
            self.assertEqual(
                dashboard.map_repo_to_preset_alias("locallama-qwen"), "qwen-profile"
            )
            # 7. Alias normalization: query has prefix, alias has no prefix
            self.assertEqual(
                dashboard.map_repo_to_preset_alias("unsloth/local-qwen"), "qwen-profile"
            )
            # 8. Single alias with prefix
            self.assertEqual(
                dashboard.map_repo_to_preset_alias("unsloth/plain-alias"),
                "plain-profile",
            )

    def test_dashboard_normalize_repo_id_and_repo_id_matches(self):
        self.assertEqual(dashboard._normalize_repo_id(None), "")
        self.assertEqual(dashboard._normalize_repo_id(123), "")
        self.assertEqual(dashboard._normalize_repo_id(""), "")
        self.assertEqual(dashboard._normalize_repo_id("unsloth/test"), "test")

        self.assertFalse(dashboard._repo_id_matches(None, "foo"))
        self.assertFalse(dashboard._repo_id_matches("foo", None))
        self.assertFalse(dashboard._repo_id_matches(123, "foo"))
        self.assertFalse(dashboard._repo_id_matches("", "foo"))
        self.assertFalse(dashboard._repo_id_matches("foo", "   "))
        self.assertTrue(dashboard._repo_id_matches("unsloth/foo", "foo"))

    def test_dashboard_map_repo_to_preset_alias_full_comma_alias(self):
        cp = configparser.ConfigParser()
        cp.read_string("""
[multi-alias-model]
alias = unsloth/alias-a, alias-b
""")
        with patch.object(dashboard, "_get_presets_config", return_value=cp):
            self.assertEqual(
                dashboard.map_repo_to_preset_alias("unsloth/alias-a, alias-b"),
                "multi-alias-model",
            )

    def test_map_repo_to_preset_alias_missing_config(self):
        with patch.object(dashboard, "_get_presets_config", return_value=None):
            self.assertEqual(
                dashboard.map_repo_to_preset_alias("unknown/model"), "unknown/model"
            )
            # Fallbacks should still work
            self.assertEqual(
                dashboard.map_repo_to_preset_alias("qwen3.6-27b-gguf:q4_k_s"),
                "Qwen3.6-27B",
            )
            self.assertEqual(
                dashboard.map_repo_to_preset_alias("qwen3.6-27b-mtp-gguf:q4_k_s"),
                "Qwen3.6-27B-spec3",
            )
            self.assertEqual(
                dashboard.map_repo_to_preset_alias("qwen3.6-35b-a3b-gguf:q4_k_s"),
                "Qwen3.6-35B-A3B",
            )
            self.assertEqual(
                dashboard.map_repo_to_preset_alias("gemma-4-test"),
                "gemma4-26a4b-routing",
            )
            self.assertIsNone(dashboard.map_repo_to_preset_alias(None))

    def test_map_repo_to_preset_alias_with_config(self):
        cp = configparser.ConfigParser()
        cp.read_string("""
[exact-model]
alias = ExactModel
hf-repo = org/exact-model

[mtp-spec-model]
alias = MTPSpec
hf-repo = org/some-repo

[base-model]
alias = BaseAlias
hf-repo = org/some-repo
""")
        with patch.object(dashboard, "_get_presets_config", return_value=cp):
            # Exact section match
            self.assertEqual(
                dashboard.map_repo_to_preset_alias("exact-model"), "exact-model"
            )
            # Substring alias match
            self.assertEqual(
                dashboard.map_repo_to_preset_alias("prefix/ExactModel-extra"),
                "exact-model",
            )
            # Repo match with mtp/spec condition
            self.assertEqual(
                dashboard.map_repo_to_preset_alias("org/some-repo-mtp"),
                "mtp-spec-model",
            )
            self.assertEqual(
                dashboard.map_repo_to_preset_alias("org/some-repo-standard"),
                "base-model",
            )
            # Unmatched returns input
            self.assertEqual(
                dashboard.map_repo_to_preset_alias("unknown-other"), "unknown-other"
            )

    def test_get_preset_metadata_missing_config(self):
        with patch.object(dashboard, "_get_presets_config", return_value=None):
            meta = dashboard.get_preset_metadata("any-model")
            self.assertEqual(meta["parallel"], "1")
            self.assertEqual(meta["flash_attn"], "true")
            self.assertEqual(meta["spec_type"], "None")

    def test_get_preset_metadata_with_config(self):
        cp = configparser.ConfigParser()
        cp.read_string("""
[*]
flash-attn = false
global-param = global_val

[custom-model]
parallel = 4
spec-type = draft
""")
        with patch.object(dashboard, "_get_presets_config", return_value=cp):
            meta = dashboard.get_preset_metadata("custom-model")
            self.assertEqual(meta["flash_attn"], "false")
            self.assertEqual(meta["global_param"], "global_val")
            self.assertEqual(meta["parallel"], "4")
            self.assertEqual(meta["spec_type"], "draft")
            # Default retained if not overridden
            self.assertEqual(meta["n_gpu_layers"], "99")


class TestFmtNum(unittest.TestCase):
    def setUp(self):
        # Since pandas is mocked, we need to explicitly set return_value for pd.isna
        dashboard.pd.isna.return_value = False

    def test_fmt_num_valid_numbers(self):
        self.assertEqual(dashboard.fmt_num(10), "10.00")
        self.assertEqual(dashboard.fmt_num(10.5), "10.50")
        self.assertEqual(dashboard.fmt_num("10.5"), "10.50")
        self.assertEqual(dashboard.fmt_num(10.556), "10.56")
        self.assertEqual(dashboard.fmt_num(10, fmt="{:.1f}"), "10.0")

    def test_fmt_num_invalid_numbers(self):
        self.assertEqual(dashboard.fmt_num("invalid"), "invalid")
        self.assertEqual(dashboard.fmt_num([1, 2]), "[1, 2]")
        self.assertEqual(dashboard.fmt_num([]), "[]")
        self.assertEqual(dashboard.fmt_num({"a": 1}), "{'a': 1}")

    def test_fmt_num_na_values(self):
        self.assertEqual(dashboard.fmt_num(None), "N/A")
        self.assertEqual(dashboard.fmt_num("N/A"), "N/A")
        self.assertEqual(dashboard.fmt_num(float("nan")), "N/A")

    def test_fmt_num_pd_na_eval(self):
        dashboard.pd.isna.return_value = True
        self.assertEqual(dashboard.fmt_num(123), "N/A")
        dashboard.pd.isna.return_value = False

    def test_fmt_num_array_ambiguity_protection(self):
        class DummyArray:
            def __bool__(self):
                raise ValueError(
                    "The truth value of an array with more than one element is ambiguous."
                )

        dashboard.pd.isna.return_value = DummyArray()
        self.assertEqual(dashboard.fmt_num([1, 2]), "[1, 2]")
        dashboard.pd.isna.return_value = False

    def test_fmt_num_booleans(self):
        self.assertEqual(dashboard.fmt_num(True), "True")
        self.assertEqual(dashboard.fmt_num(False), "False")

    def test_fmt_num_infinity(self):
        self.assertEqual(dashboard.fmt_num(float("inf")), "Inf")
        self.assertEqual(dashboard.fmt_num(-float("inf")), "-Inf")


class TestExtractReasoningAccData(unittest.TestCase):
    def test_extract_reasoning_acc_data_normal(self):
        class MockColumns(list):
            def get_loc(self, key):
                return self.index(key)

        class MockDataFrame:
            empty = False
            columns = MockColumns(["Model", "KV Quant", "Needle", "RULER", "Other"])

            def itertuples(self, index=False, name=None):
                return [
                    ("Model-A", "q4_k_m", "Pass", "Fail", 100),
                    ("Model-B", "q8_0", "Pass", "N/A", 200),
                ]

        res = dashboard.extract_reasoning_acc_data(MockDataFrame())
        self.assertEqual(len(res), 3)
        self.assertEqual(
            res[0],
            {"Model_Quant": "Model-A (q4_k_m)", "Test Suite": "Needle", "Score": 1.0},
        )
        self.assertEqual(
            res[1],
            {"Model_Quant": "Model-A (q4_k_m)", "Test Suite": "RULER", "Score": 0.0},
        )
        self.assertEqual(
            res[2],
            {"Model_Quant": "Model-B (q8_0)", "Test Suite": "Needle", "Score": 1.0},
        )

    def test_extract_reasoning_acc_data_missing_required_columns(self):
        class MockColumns(list):
            def get_loc(self, key):
                return self.index(key)

        class MockDataFrame:
            empty = False
            columns = MockColumns(["Model", "Needle"])

            def itertuples(self, index=False, name=None):
                return [("Model-A", "Pass")]

        self.assertEqual(dashboard.extract_reasoning_acc_data(MockDataFrame()), [])
        self.assertEqual(dashboard.extract_reasoning_acc_data(None), [])

        class EmptyDF:
            empty = True
            columns = MockColumns(["Model", "KV Quant", "Needle"])

        self.assertEqual(dashboard.extract_reasoning_acc_data(EmptyDF()), [])

    def test_extract_reasoning_acc_data_no_test_suites(self):
        class MockColumns(list):
            def get_loc(self, key):
                return self.index(key)

        class MockDataFrame:
            empty = False
            columns = MockColumns(["Model", "KV Quant", "Throughput"])

            def itertuples(self, index=False, name=None):
                return [("Model-A", "q4", 50.0)]

        self.assertEqual(dashboard.extract_reasoning_acc_data(MockDataFrame()), [])

    def test_extract_reasoning_acc_data_duplicate_columns(self):
        class MockColumns(list):
            def get_loc(self, key):
                if key == "Model":
                    return [True, False, False]
                return self.index(key)

        class MockDataFrame:
            empty = False
            columns = MockColumns(["Model", "KV Quant", "Needle"])

            def itertuples(self, index=False, name=None):
                return [("Model-A", "q4", "Pass")]

        res = dashboard.extract_reasoning_acc_data(MockDataFrame())
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["Model_Quant"], "Model-A (q4)")


class TestLoadRuns(unittest.TestCase):
    def setUp(self):
        dashboard.st.error.reset_mock()
        dashboard.load_runs.clear()

    def tearDown(self):
        dashboard.load_runs.clear()

    def test_load_runs_caching_and_clear(self):
        with patch.object(dashboard.Path, "glob") as mock_glob:
            mock_glob.return_value = []

            # First call executes glob
            res1 = dashboard.load_runs()
            self.assertEqual(mock_glob.call_count, 1)

            # Second call hits cache without calling glob
            res2 = dashboard.load_runs()
            self.assertEqual(mock_glob.call_count, 1)
            self.assertIs(res1, res2)

            # Calling clear() busts the cache
            dashboard.load_runs.clear()
            res3 = dashboard.load_runs()
            self.assertEqual(mock_glob.call_count, 2)

    def test_load_runs_ttl_expiration(self):
        with patch.object(dashboard.Path, "glob") as mock_glob:
            mock_glob.return_value = []

            current_time = 1000.0
            with patch.object(dashboard.time, "time", side_effect=lambda: current_time):
                dashboard.load_runs()
                self.assertEqual(mock_glob.call_count, 1)

                # Advance time by 30 seconds (still valid, ttl=60)
                current_time = 1030.0
                dashboard.load_runs()
                self.assertEqual(mock_glob.call_count, 1)

                # Advance time by 61 seconds (expired, should re-query)
                current_time = 1061.0
                dashboard.load_runs()
                self.assertEqual(mock_glob.call_count, 2)

    def test_load_runs_corrupted_json_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            valid_run = tmp_path / "run_2026-01-01T00-00-00.json"
            valid_run.write_text(
                json.dumps(
                    {
                        "run_metadata": {
                            "timestamp": "2026-01-01T00:00:00",
                            "target_endpoint": "http://localhost:8080",
                        },
                        "model_settings": {
                            "profile_alias": "test-model",
                            "base_quantization": "Q4_K_S",
                        },
                        "throughput_metrics": {
                            "prefill_speed": 120.0,
                            "decode_speed": 35.0,
                        },
                        "reasoning_accuracy": {"needle": "Pass"},
                        "quantization_loss": {"perplexity": 5.4},
                    }
                )
            )

            corrupted_run = tmp_path / "run_corrupted.json"
            corrupted_run.write_text("{invalid json: error")

            old_dir = dashboard.HISTORY_DIR
            dashboard.HISTORY_DIR = tmp_path
            try:
                with patch.object(dashboard.pd, "DataFrame") as mock_df_ctor:
                    mock_df = MagicMock()
                    mock_df.empty = False
                    mock_df_ctor.return_value = mock_df

                    res = dashboard.load_runs()
                    self.assertIs(res, mock_df.sort_values().reset_index())

                    # Verify st.error was called for the corrupted file
                    dashboard.st.error.assert_called()
                    error_msg = dashboard.st.error.call_args[0][0]
                    self.assertIn("run_corrupted.json", error_msg)

                    # Verify pd.DataFrame was called with the valid run
                    self.assertEqual(mock_df_ctor.call_count, 1)
                    parsed_runs = mock_df_ctor.call_args[0][0]
                    self.assertEqual(len(parsed_runs), 1)
                    self.assertEqual(
                        parsed_runs[0]["Filename"], "run_2026-01-01T00-00-00.json"
                    )
                    self.assertEqual(parsed_runs[0]["Model"], "test-model")
                    self.assertEqual(parsed_runs[0]["Base Quant"], "Q4_K_S")
                    self.assertEqual(parsed_runs[0]["Prefill (t/s)"], 120.0)
            finally:
                dashboard.HISTORY_DIR = old_dir

    def test_load_runs_return_type_and_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir = dashboard.HISTORY_DIR
            dashboard.HISTORY_DIR = Path(tmpdir)
            try:
                with patch.object(dashboard.pd, "DataFrame") as mock_df_ctor:
                    mock_df = MagicMock()
                    mock_df.empty = True
                    mock_df_ctor.return_value = mock_df

                    res = dashboard.load_runs()
                    self.assertIs(res, mock_df)
                    mock_df_ctor.assert_called_once_with([])
            finally:
                dashboard.HISTORY_DIR = old_dir

    def test_load_runs_complex_parsing_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            run_file = tmp_path / "run_complex.json"
            run_file.write_text(
                json.dumps(
                    {
                        "run_metadata": {
                            "timestamp": "2026-09-11T00:00:00",
                            "target_endpoint": "http://localhost:8080",
                            "cli_arguments": ["--tokens", "65536"],
                        },
                        "model_settings": {
                            "model_name": "/models/qwen-spec4-UD-Q4_K_S.gguf",
                            "base_quantization": "Unknown",
                            "kv_cache_quant": "q4_0",
                            "threads": 8,
                            "ubatch_size": 512,
                            "batch_size": 2048,
                            "spec_type": "draft",
                            "spec_draft_type_k": "q4_0",
                            "spec_draft_type_v": "q4_0",
                            "flash_attn": "true",
                            "parallel": "2",
                            "fit": "true",
                        },
                        "throughput_metrics": {
                            "prefill_speed": 150.0,
                            "decode_speed": 40.0,
                            "ttft": 0.12,
                        },
                        "reasoning_accuracy": {
                            "needle": "Pass",
                            "ruler": 0.95,
                            "longbench": 0.88,
                            "swe_bench": "N/A",
                        },
                        "quantization_loss": {
                            "perplexity": 4.5,
                            "mean_kld": 0.02,
                            "same_top_match_percent": 98.5,
                        },
                    }
                )
            )

            old_dir = dashboard.HISTORY_DIR
            dashboard.HISTORY_DIR = tmp_path
            try:
                with patch.object(dashboard.pd, "DataFrame") as mock_df_ctor:
                    mock_df = MagicMock()
                    mock_df.empty = False
                    mock_df_ctor.return_value = mock_df

                    res = dashboard.load_runs()
                    self.assertIs(res, mock_df.sort_values().reset_index())
                    self.assertEqual(mock_df_ctor.call_count, 1)
                    parsed_runs = mock_df_ctor.call_args[0][0]
                    self.assertEqual(len(parsed_runs), 1)
                    r = parsed_runs[0]
                    self.assertEqual(r["Context Length"], 65536)
                    self.assertEqual(r["Base Quant"], "Q4_K_S")
                    self.assertEqual(r["Threads"], 8)
                    self.assertEqual(r["Ubatch Size"], 512)
                    self.assertEqual(r["Batch Size"], 2048)
            finally:
                dashboard.HISTORY_DIR = old_dir

    def test_is_mock_utility(self):
        self.assertTrue(dashboard._is_mock(MagicMock()))
        self.assertTrue(dashboard._is_mock(unittest.mock.Mock()))
        self.assertTrue(dashboard._is_mock(unittest.mock.NonCallableMagicMock()))

        class CustomMockLike:
            _mock_return_value = True

        self.assertTrue(dashboard._is_mock(CustomMockLike()))

        def regular_func():
            pass

        self.assertFalse(dashboard._is_mock(regular_func))
        self.assertFalse(dashboard._is_mock(123))
        self.assertFalse(dashboard._is_mock("string"))

    def test_fallback_cache_data_direct(self):
        call_count = 0

        @dashboard._fallback_cache_data
        def simple_cached(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        self.assertEqual(simple_cached(5), 10)
        self.assertEqual(simple_cached(5), 10)
        self.assertEqual(call_count, 1)

        simple_cached.clear()
        self.assertEqual(simple_cached(5), 10)
        self.assertEqual(call_count, 2)

    def test_fallback_cache_data_with_ttl(self):
        call_count = 0
        current_time = 100.0

        with patch.object(dashboard.time, "time", side_effect=lambda: current_time):

            @dashboard._fallback_cache_data(ttl=10)
            def simple_cached_ttl(x):
                nonlocal call_count
                call_count += 1
                return x * 3

            self.assertEqual(simple_cached_ttl(4), 12)
            self.assertEqual(simple_cached_ttl(4), 12)
            self.assertEqual(call_count, 1)

            # Advance by 5s (still valid)
            current_time = 105.0
            self.assertEqual(simple_cached_ttl(4), 12)
            self.assertEqual(call_count, 1)

            # Advance by 11s (expired)
            current_time = 111.0
            self.assertEqual(simple_cached_ttl(4), 12)
            self.assertEqual(call_count, 2)


class TestParseRunFile(unittest.TestCase):
    def setUp(self):
        dashboard.st.error.reset_mock()

    def test_parse_run_file_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "run_2026-09-11T12-00-00.json"
            data = {
                "run_metadata": {
                    "timestamp": "2026-09-11T12:00:00",
                    "target_endpoint": "http://127.0.0.1:8081",
                    "cli_arguments": ["--tokens", "131072"],
                },
                "model_settings": {
                    "profile_alias": "qwen-27b",
                    "base_quantization": "Q4_K_M",
                    "kv_cache_quant": "q8_0",
                    "threads": 12,
                    "ubatch_size": 256,
                    "batch_size": 1024,
                    "spec_type": "eagle",
                    "spec_draft_type_k": "q4_0",
                    "spec_draft_type_v": "q4_0",
                    "flash_attn": "true",
                    "parallel": "2",
                    "fit": "false",
                },
                "throughput_metrics": {
                    "prefill_speed": 180.5,
                    "decode_speed": 45.2,
                    "ttft": 0.08,
                },
                "reasoning_accuracy": {
                    "needle": "Pass",
                    "ruler": 0.96,
                    "longbench": 0.89,
                    "swe_bench": "Pass",
                },
                "quantization_loss": {
                    "perplexity": 4.12,
                    "mean_kld": 0.015,
                    "same_top_match_percent": 99.2,
                },
            }
            file_path.write_text(json.dumps(data), encoding="utf-8")

            res = dashboard._parse_run_file(file_path)
            self.assertIsNotNone(res)
            self.assertEqual(res["Filename"], "run_2026-09-11T12-00-00.json")
            self.assertEqual(res["Timestamp"], "2026-09-11T12:00:00")
            self.assertEqual(res["Endpoint"], "http://127.0.0.1:8081")
            self.assertEqual(res["Model"], "qwen-27b")
            self.assertEqual(res["Base Quant"], "Q4_K_M")
            self.assertEqual(res["KV Quant"], "q8_0")
            self.assertEqual(res["Threads"], 12)
            self.assertEqual(res["Ubatch Size"], 256)
            self.assertEqual(res["Batch Size"], 1024)
            self.assertEqual(res["Speculative"], "eagle")
            self.assertEqual(res["Spec Type"], "eagle")
            self.assertEqual(res["Spec Draft Type K"], "q4_0")
            self.assertEqual(res["Spec Draft Type V"], "q4_0")
            self.assertEqual(res["Flash Attn"], "true")
            self.assertEqual(res["Parallel"], "2")
            self.assertEqual(res["Fit"], "false")
            self.assertEqual(res["Prefill (t/s)"], 180.5)
            self.assertEqual(res["Decode (t/s)"], 45.2)
            self.assertEqual(res["TTFT (s)"], 0.08)
            self.assertEqual(res["Needle"], "Pass")
            self.assertEqual(res["RULER"], 0.96)
            self.assertEqual(res["LongBench"], 0.89)
            self.assertEqual(res["SWE-bench"], "Pass")
            self.assertEqual(res["PPL"], 4.12)
            self.assertEqual(res["KLD"], 0.015)
            self.assertEqual(res["Same Top %"], 99.2)
            self.assertEqual(res["Context Length"], 131072)

    def test_parse_run_file_malformed_json_and_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Corrupted JSON content
            corrupt_file = tmp / "run_corrupt.json"
            corrupt_file.write_text("{invalid json: error", encoding="utf-8")
            res = dashboard._parse_run_file(corrupt_file)
            self.assertIsNone(res)
            dashboard.st.error.assert_called()

            # Non-existent file
            dashboard.st.error.reset_mock()
            res = dashboard._parse_run_file(tmp / "non_existent.json")
            self.assertIsNone(res)
            dashboard.st.error.assert_called()

            # Root is a list rather than a dict
            dashboard.st.error.reset_mock()
            list_file = tmp / "run_list.json"
            list_file.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            res = dashboard._parse_run_file(list_file)
            self.assertIsNone(res)
            dashboard.st.error.assert_called()

    def test_parse_run_file_missing_fields_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "run_empty.json"
            file_path.write_text("{}", encoding="utf-8")

            res = dashboard._parse_run_file(file_path)
            self.assertIsNotNone(res)
            self.assertEqual(res["Filename"], "run_empty.json")
            self.assertEqual(res["Timestamp"], "Unknown")
            self.assertEqual(res["Endpoint"], "Unknown")
            self.assertEqual(res["Context Length"], 200000)
            self.assertEqual(res["Base Quant"], "Q4_K_S")
            self.assertEqual(res["KV Quant"], "Unknown")
            self.assertIsNone(res["Threads"])
            self.assertEqual(res["Needle"], "N/A")
            self.assertEqual(res["RULER"], "N/A")
            self.assertEqual(res["LongBench"], "N/A")
            self.assertEqual(res["SWE-bench"], "N/A")
            self.assertIsNone(res["Prefill (t/s)"])
            self.assertIsNone(res["PPL"])
            self.assertIsNone(res["KLD"])
            self.assertEqual(res["Flash Attn"], "true")
            self.assertEqual(res["Parallel"], "1")
            self.assertIn(res["Fit"], ("true", "false"))

    def test_parse_run_file_invalid_tokens_arg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "run_invalid_tokens.json"
            data = {"run_metadata": {"cli_arguments": ["--tokens", "not-a-number"]}}
            file_path.write_text(json.dumps(data), encoding="utf-8")
            res = dashboard._parse_run_file(file_path)
            self.assertIsNotNone(res)
            self.assertEqual(res["Context Length"], 200000)

    def test_parse_run_file_raw_gguf_cleaning(self):
        cases = [
            ("/models/custom-model-UD-Q4_K_XL.gguf", "custom-model"),
            ("/models/custom-model-UD-Q4_K_S.gguf", "custom-model"),
            ("custom-model-Q4_K_S.gguf", "custom-model"),
            ("custom-model-Q6_K_XL.gguf", "custom-model"),
            ("custom-model-Q8_0.gguf", "custom-model"),
            ("custom-model-F16.gguf", "custom-model"),
            ("custom-model-UD.gguf", "custom-model"),
        ]
        for raw_name, expected_cleaned in cases:
            with (
                self.subTest(raw_name=raw_name),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                file_path = Path(tmpdir) / "run_test.json"
                data = {"model_settings": {"profile_alias": raw_name}}
                file_path.write_text(json.dumps(data), encoding="utf-8")
                res = dashboard._parse_run_file(file_path)
                self.assertIsNotNone(res)
                self.assertEqual(res["Model"], expected_cleaned)

    def test_parse_run_file_quantization_resolutions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "run_quant.json"

            # 1. Explicit base_quantization provided
            file_path.write_text(
                json.dumps({"model_settings": {"base_quantization": "Q5_K_M"}}),
                encoding="utf-8",
            )
            res = dashboard._parse_run_file(file_path)
            self.assertEqual(res["Base Quant"], "Q5_K_M")

            # 2. Unknown base_quantization, model_name contains colon
            file_path.write_text(
                json.dumps(
                    {
                        "model_settings": {
                            "base_quantization": "Unknown",
                            "model_name": "repo/model:Q8_0",
                        }
                    }
                ),
                encoding="utf-8",
            )
            res = dashboard._parse_run_file(file_path)
            self.assertEqual(res["Base Quant"], "Q8_0")

            # 3. Unknown base_quantization, model_name contains quant alias (e.g. q5_k_m or q8_0)
            file_path.write_text(
                json.dumps(
                    {
                        "model_settings": {
                            "base_quantization": "Unknown",
                            "model_name": "qwen2.5-coder-q5_k_m.gguf",
                        }
                    }
                ),
                encoding="utf-8",
            )
            res = dashboard._parse_run_file(file_path)
            self.assertEqual(res["Base Quant"], "Q5_K_M")

            # 4. Unknown base_quantization, no colon/alias, but profile_name has spec4 -> Q6_K_XL
            file_path.write_text(
                json.dumps(
                    {
                        "model_settings": {
                            "base_quantization": "Unknown",
                            "profile_alias": "local-spec4-test",
                        }
                    }
                ),
                encoding="utf-8",
            )
            res = dashboard._parse_run_file(file_path)
            self.assertEqual(res["Base Quant"], "Q6_K_XL")

            # 5. Unknown base_quantization, fallback default -> Q4_K_S
            file_path.write_text(
                json.dumps(
                    {
                        "model_settings": {
                            "base_quantization": "Unknown",
                            "profile_alias": "standard-model",
                        }
                    }
                ),
                encoding="utf-8",
            )
            res = dashboard._parse_run_file(file_path)
            self.assertEqual(res["Base Quant"], "Q4_K_S")

    def test_parse_run_file_speculative_resolution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "run_spec.json"

            # Speculative type from settings spec_type
            file_path.write_text(
                json.dumps(
                    {
                        "model_settings": {
                            "spec_type": "draft",
                            "spec_draft_type_k": "q8_0",
                            "spec_draft_type_v": "q8_0",
                        }
                    }
                ),
                encoding="utf-8",
            )
            res = dashboard._parse_run_file(file_path)
            self.assertEqual(res["Spec Type"], "draft")
            self.assertEqual(res["Spec Draft Type K"], "q8_0")
            self.assertEqual(res["Spec Draft Type V"], "q8_0")

            # Speculative type fallback to speculative_draft_type
            file_path.write_text(
                json.dumps(
                    {
                        "model_settings": {
                            "speculative_draft_type": "spec_draft_fallback",
                            "spec_draft_type_k": "None",
                        }
                    }
                ),
                encoding="utf-8",
            )
            res = dashboard._parse_run_file(file_path)
            self.assertEqual(res["Spec Type"], "spec_draft_fallback")


class TestDashboardValidators(unittest.TestCase):
    def test_validate_gguf_path_none_or_empty(self):
        self.assertIsNone(dashboard.validate_gguf_path(None))
        self.assertEqual(dashboard.validate_gguf_path(""), "")

    def test_validate_gguf_path_existing_allowed_file(self):
        # Tempfile inside cwd
        with tempfile.NamedTemporaryFile(dir=Path.cwd(), suffix=".gguf") as tmp:
            self.assertEqual(
                dashboard.validate_gguf_path(tmp.name), str(Path(tmp.name).resolve())
            )

        # Relative path inside cwd
        with tempfile.NamedTemporaryFile(
            dir=Path.cwd(), prefix="test_gguf_", suffix=".gguf"
        ) as tmp:
            rel_name = os.path.basename(tmp.name)
            self.assertEqual(
                dashboard.validate_gguf_path(rel_name), str(Path(tmp.name).resolve())
            )

        # Tempfile inside Path(__file__).parent
        with tempfile.NamedTemporaryFile(
            dir=Path(__file__).parent.resolve(), suffix=".gguf"
        ) as tmp:
            self.assertEqual(
                dashboard.validate_gguf_path(tmp.name), str(Path(tmp.name).resolve())
            )

        # Tempfiles inside patched Path.home() .cache and models directories
        with tempfile.TemporaryDirectory() as fake_home:
            cache_dir = Path(fake_home) / ".cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            models_dir = Path(fake_home) / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            with patch.object(Path, "home", return_value=Path(fake_home)):
                with tempfile.NamedTemporaryFile(
                    dir=cache_dir, suffix=".gguf"
                ) as tmp_cache:
                    self.assertEqual(
                        dashboard.validate_gguf_path(tmp_cache.name),
                        str(Path(tmp_cache.name).resolve()),
                    )
                with tempfile.NamedTemporaryFile(
                    dir=models_dir, suffix=".gguf"
                ) as tmp_models:
                    self.assertEqual(
                        dashboard.validate_gguf_path(tmp_models.name),
                        str(Path(tmp_models.name).resolve()),
                    )

    def test_validate_gguf_path_symlink(self):
        with tempfile.NamedTemporaryFile(dir=Path.cwd(), suffix=".gguf") as target:
            symlink = Path.cwd() / f"test_symlink_{os.path.basename(target.name)}"
            try:
                symlink.symlink_to(target.name)
                self.assertEqual(
                    dashboard.validate_gguf_path(str(symlink)),
                    str(Path(target.name).resolve()),
                )
            finally:
                if symlink.is_symlink() or symlink.exists():
                    symlink.unlink(missing_ok=True)

    def test_validate_gguf_path_nonexistent_file(self):
        with self.assertRaisesRegex(ValueError, r"GGUF file path does not exist"):
            dashboard.validate_gguf_path("non_existent_file.gguf")
        with self.assertRaisesRegex(ValueError, r"GGUF file path does not exist"):
            dashboard.validate_gguf_path("/path/to/nowhere/model.gguf")

    def test_validate_gguf_path_directory(self):
        with self.assertRaisesRegex(ValueError, r"GGUF path is not a file"):
            dashboard.validate_gguf_path(str(Path.cwd()))
        with self.assertRaisesRegex(ValueError, r"GGUF path is not a file"):
            dashboard.validate_gguf_path(".")

    def test_validate_gguf_path_escapes_allowed_parents(self):
        # Deterministically trigger escapes using temp directories and mocked allowed parents
        with tempfile.TemporaryDirectory() as mock_parent_dir:
            with tempfile.TemporaryDirectory() as outside_dir:
                outside_file = Path(outside_dir) / "escaped.gguf"
                outside_file.write_text("dummy model content")
                with (
                    patch.object(Path, "cwd", return_value=Path(mock_parent_dir)),
                    patch.object(Path, "home", return_value=Path(mock_parent_dir)),
                ):
                    with self.assertRaisesRegex(
                        ValueError, r"GGUF path escapes allowed parent directories"
                    ):
                        dashboard.validate_gguf_path(str(outside_file))

    def test_validate_corpus_name_valid(self):
        self.assertEqual(
            dashboard.validate_corpus_name("kld_corpus.txt"), "kld_corpus.txt"
        )
        self.assertEqual(dashboard.validate_corpus_name("corpus.json"), "corpus.json")
        self.assertEqual(dashboard.validate_corpus_name("custom_eval"), "custom_eval")
        # Leading and trailing whitespace should be stripped
        self.assertEqual(
            dashboard.validate_corpus_name("  kld_corpus.txt  "), "kld_corpus.txt"
        )

    def test_validate_corpus_name_empty_or_none(self):
        with self.assertRaisesRegex(ValueError, r"Corpus name cannot be empty\."):
            dashboard.validate_corpus_name("")
        with self.assertRaisesRegex(ValueError, r"Corpus name cannot be empty\."):
            dashboard.validate_corpus_name(None)
        with self.assertRaisesRegex(ValueError, r"Corpus name cannot be empty\."):
            dashboard.validate_corpus_name("   ")
        with self.assertRaisesRegex(ValueError, r"Corpus name cannot be empty\."):
            dashboard.validate_corpus_name(" \t \n ")

    def test_validate_corpus_name_dot_or_dotdot(self):
        with self.assertRaisesRegex(ValueError, r"Invalid corpus name"):
            dashboard.validate_corpus_name(".")
        with self.assertRaisesRegex(ValueError, r"Invalid corpus name"):
            dashboard.validate_corpus_name("..")
        with self.assertRaisesRegex(ValueError, r"Invalid corpus name"):
            dashboard.validate_corpus_name("./")
        with self.assertRaisesRegex(ValueError, r"Invalid corpus name"):
            dashboard.validate_corpus_name("../")
        with self.assertRaisesRegex(ValueError, r"Invalid corpus name"):
            dashboard.validate_corpus_name("/")
        with self.assertRaisesRegex(ValueError, r"Invalid corpus name"):
            dashboard.validate_corpus_name(" / ")

    def test_validate_corpus_name_directory_path(self):
        self.assertEqual(
            dashboard.validate_corpus_name("/path/to/kld_corpus.txt"), "kld_corpus.txt"
        )
        self.assertEqual(
            dashboard.validate_corpus_name("corpora/nested/dataset.csv"), "dataset.csv"
        )
        self.assertEqual(
            dashboard.validate_corpus_name("./local/dir/test_corpus"), "test_corpus"
        )

    def test_validate_gguf_path_rejects_non_gguf(self):
        # Non-.gguf files in cwd
        for ext in [".txt", ".py", ".bin", ".json", "", ".dat"]:
            with tempfile.NamedTemporaryFile(dir=Path.cwd(), suffix=ext) as tmp:
                with self.assertRaisesRegex(
                    ValueError, r"GGUF file must have a \.gguf extension"
                ):
                    dashboard.validate_gguf_path(tmp.name)

        # Case-insensitive: uppercase .GGUF is accepted
        with tempfile.NamedTemporaryFile(dir=Path.cwd(), suffix=".GGUF") as tmp:
            self.assertEqual(
                dashboard.validate_gguf_path(tmp.name), str(Path(tmp.name).resolve())
            )

    def test_validate_gguf_path_home_root_rejection_and_allowed_subdirs(self):
        with tempfile.TemporaryDirectory() as fake_home:
            # Home root non-.gguf file (e.g. ~/.bashrc)
            bashrc = Path(fake_home) / ".bashrc"
            bashrc.write_text("export TEST=1")

            # Home root .gguf file
            root_gguf = Path(fake_home) / "root_model.gguf"
            root_gguf.write_text("dummy gguf")

            # Subdir .cache
            cache_dir = Path(fake_home) / ".cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_gguf = cache_dir / "cached_model.gguf"
            cache_gguf.write_text("dummy cached gguf")

            # Subdir models
            models_dir = Path(fake_home) / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            models_gguf = models_dir / "model_in_models.gguf"
            models_gguf.write_text("dummy models gguf")

            # Nested subdir inside .cache (e.g. huggingface hub)
            nested_cache_dir = cache_dir / "huggingface" / "hub"
            nested_cache_dir.mkdir(parents=True, exist_ok=True)
            nested_cache_gguf = nested_cache_dir / "nested_model.gguf"
            nested_cache_gguf.write_text("dummy nested cached gguf")

            with patch.object(Path, "home", return_value=Path(fake_home)):
                # Rejects ~/.bashrc due to non-.gguf extension
                with self.assertRaisesRegex(
                    ValueError, r"GGUF file must have a \.gguf extension"
                ):
                    dashboard.validate_gguf_path(str(bashrc))

                # Rejects root_model.gguf in home root because it escapes allowed parent directories
                with self.assertRaisesRegex(
                    ValueError, r"GGUF path escapes allowed parent directories"
                ):
                    dashboard.validate_gguf_path(str(root_gguf))

                # Accepts .gguf in ~/.cache
                self.assertEqual(
                    dashboard.validate_gguf_path(str(cache_gguf)),
                    str(cache_gguf.resolve()),
                )

                # Accepts .gguf in nested ~/.cache
                self.assertEqual(
                    dashboard.validate_gguf_path(str(nested_cache_gguf)),
                    str(nested_cache_gguf.resolve()),
                )

                # Accepts .gguf in ~/models
                self.assertEqual(
                    dashboard.validate_gguf_path(str(models_gguf)),
                    str(models_gguf.resolve()),
                )

    def test_validate_new_tokens_valid(self):
        self.assertEqual(dashboard.validate_new_tokens(1), 1)
        self.assertEqual(dashboard.validate_new_tokens(5000), 5000)
        self.assertEqual(dashboard.validate_new_tokens(262144), 262144)
        self.assertEqual(dashboard.validate_new_tokens("5000"), 5000)
        self.assertEqual(dashboard.validate_new_tokens(5000.0), 5000)

    def test_validate_new_tokens_out_of_bounds(self):
        for out_val in [0, -1, -5000, 262145, 1000000]:
            with self.subTest(val=out_val):
                with self.assertRaisesRegex(
                    ValueError, r"Context length tokens must be between 1 and 262144\."
                ):
                    dashboard.validate_new_tokens(out_val)

    def test_validate_new_tokens_invalid_type(self):
        for bad_val in [
            None,
            True,
            False,
            "abc",
            "",
            "12.34",
            12.34,
            [5000],
            {"tokens": 5000},
        ]:
            with self.subTest(val=bad_val):
                with self.assertRaisesRegex(
                    ValueError, r"Context length tokens must be between 1 and 262144\."
                ):
                    dashboard.validate_new_tokens(bad_val)


import importlib.util

HAS_PANDAS_AND_PLOTLY = (
    importlib.util.find_spec("pandas") is not None
    and importlib.util.find_spec("plotly") is not None
)


@unittest.skipUnless(
    HAS_PANDAS_AND_PLOTLY, "pandas and plotly required for throughput figure tests"
)
class TestBuildThroughputFigure(unittest.TestCase):
    def setUp(self):
        import plotly.graph_objects as real_go

        self._orig_go = dashboard.go
        dashboard.go = real_go

    def tearDown(self):
        dashboard.go = self._orig_go

    def test_empty_dataframe(self):
        import pandas as pd

        df = pd.DataFrame(
            columns=[
                "Model",
                "KV Quant",
                "Context Length",
                "Prefill (t/s)",
                "Decode (t/s)",
            ]
        )
        fig = dashboard.build_throughput_figure(df)
        self.assertEqual(len(fig.data), 0)

    def test_bare_empty_dataframe(self):
        import pandas as pd

        fig = dashboard.build_throughput_figure(pd.DataFrame())
        self.assertEqual(len(fig.data), 0)

    def test_none_input(self):
        fig = dashboard.build_throughput_figure(None)
        self.assertEqual(len(fig.data), 0)

    def test_missing_schema_columns(self):
        import pandas as pd

        fig = dashboard.build_throughput_figure(pd.DataFrame({"Model": ["M1"]}))
        self.assertEqual(len(fig.data), 0)

    def test_non_dataframe_input(self):
        self.assertEqual(
            len(dashboard.build_throughput_figure("invalid_string").data), 0
        )
        self.assertEqual(len(dashboard.build_throughput_figure([1, 2, 3]).data), 0)
        self.assertEqual(len(dashboard.build_throughput_figure(123).data), 0)

    def test_missing_and_nan_quants(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "Model": ["M1", "M1"],
                "KV Quant": [None, float("nan")],
                "Context Length": [1024, 2048],
                "Prefill (t/s)": [100.0, 110.0],
                "Decode (t/s)": [30.0, 32.0],
            }
        )
        fig = dashboard.build_throughput_figure(df)
        self.assertEqual(len(fig.data), 0)

    def test_multiple_models_and_quants_and_ordering(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "Model": [
                    "Model_B",
                    "Model_A",
                    "Model_A",
                    "Model_A",
                    "Model_A",
                    "Model_Unknown",
                ],
                "KV Quant": ["f16", "q8_0", "f16", "q5_1", "q4_0", "custom_quant"],
                "Context Length": [4096, 2048, 1024, 1024, 512, 128],
                "Prefill (t/s)": [100.0, 120.0, 150.0, 140.0, 160.0, 90.0],
                "Decode (t/s)": [30.0, 35.0, 45.0, 42.0, 48.0, 25.0],
            }
        )
        fig = dashboard.build_throughput_figure(df)
        # 6 (model, quant) groups * 2 traces each (PP, TG) = 12 traces
        self.assertEqual(len(fig.data), 12)

        # Traces are ordered alphabetically by model, then quant
        # Model_A groups: f16, q4_0, q5_1, q8_0
        # Model_A f16 (PP & TG) - first quant for Model_A -> showlegend=True, solid dash
        t0 = fig.data[0]
        self.assertEqual(t0.name, "Model_A (PP)")
        self.assertTrue(t0.showlegend)
        self.assertEqual(t0.line.dash, "solid")
        self.assertEqual(tuple(t0.customdata[0]), ("Model_A", "f16"))

        t1 = fig.data[1]
        self.assertEqual(t1.name, "Model_A (TG)")
        self.assertTrue(t1.showlegend)

        # Model_A q4_0 - second quant -> showlegend=False, dashdot dash
        t2 = fig.data[2]
        self.assertEqual(t2.name, "Model_A (PP)")
        self.assertFalse(t2.showlegend)
        self.assertEqual(t2.line.dash, "dashdot")

        # Model_A q5_1 -> dot dash
        t4 = fig.data[4]
        self.assertEqual(t4.line.dash, "dot")

        # Model_A q8_0 -> dash dash
        t6 = fig.data[6]
        self.assertEqual(t6.line.dash, "dash")

        # Model_B f16 - first quant for Model_B -> showlegend=True, solid dash
        t8 = fig.data[8]
        self.assertEqual(t8.name, "Model_B (PP)")
        self.assertTrue(t8.showlegend)
        self.assertEqual(t8.line.dash, "solid")

        # Model_Unknown custom_quant - fallback color and dashdot
        t10 = fig.data[10]
        self.assertEqual(t10.name, "Model_Unknown (PP)")
        self.assertEqual(t10.marker.color, "#94a3b8")
        self.assertEqual(t10.line.dash, "dashdot")

    def test_sorting_by_context_length(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "Model": ["Model_A", "Model_A"],
                "KV Quant": ["f16", "f16"],
                "Context Length": [4096, 512],
                "Prefill (t/s)": [100.0, 150.0],
                "Decode (t/s)": [30.0, 45.0],
            }
        )
        fig = dashboard.build_throughput_figure(df)
        t0 = fig.data[0]
        self.assertEqual(list(t0.x), [512, 4096])


class TestEnqueueOutput(unittest.TestCase):
    def test_reading_multi_line_text(self):
        import io
        import queue

        text = "line 1\nline 2\nline 3\n"
        stream = io.StringIO(text)
        q = queue.Queue()

        dashboard.enqueue_output(stream, q)

        results = []
        while not q.empty():
            results.append(q.get_nowait())

        self.assertEqual(results, ["line 1\n", "line 2\n", "line 3\n"])
        self.assertTrue(stream.closed)

    def test_reading_empty_stream(self):
        import io
        import queue

        stream = io.StringIO("")
        q = queue.Queue()

        dashboard.enqueue_output(stream, q)

        self.assertTrue(q.empty())
        self.assertTrue(stream.closed)

    def test_stream_closed_on_completion_with_mock(self):
        import queue

        mock_stream = MagicMock()
        mock_stream.readline.side_effect = ["output 1\n", "output 2\n", ""]
        q = queue.Queue()

        dashboard.enqueue_output(mock_stream, q)

        results = []
        while not q.empty():
            results.append(q.get_nowait())

        self.assertEqual(results, ["output 1\n", "output 2\n"])
        mock_stream.close.assert_called_once()

    def test_stream_closed_on_exception(self):
        import queue

        mock_stream = MagicMock()
        mock_stream.readline.side_effect = RuntimeError("Read error")
        q = queue.Queue()

        with self.assertRaises(RuntimeError) as ctx:
            dashboard.enqueue_output(mock_stream, q)
        self.assertEqual(str(ctx.exception), "Read error")
        mock_stream.close.assert_called_once()

    def test_stream_close_exception_suppressed(self):
        import queue

        mock_stream = MagicMock()
        mock_stream.readline.side_effect = ["line 1\n", ""]
        mock_stream.close.side_effect = OSError("Failed to close stream")
        q = queue.Queue()

        dashboard.enqueue_output(mock_stream, q)

        self.assertEqual(q.get_nowait(), "line 1\n")
        self.assertTrue(q.empty())
        mock_stream.close.assert_called_once()


class TestModelRetrieval(unittest.TestCase):
    def test_fetch_available_models_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"id": "unsloth/Qwen3.6-27B-GGUF:Q4_K_XL"},
                {"id": "unsloth/Qwen3.6-27B-GGUF:Q4_K_S"},
                {"id": "locallama-qwen-hass"},
            ]
        }
        with patch.object(
            dashboard.requests, "get", return_value=mock_resp
        ) as mock_get:
            models = dashboard.fetch_available_models("http://127.0.0.1:8083")

        self.assertEqual(
            models,
            [
                "locallama-qwen-hass",
                "unsloth/Qwen3.6-27B-GGUF:Q4_K_S",
                "unsloth/Qwen3.6-27B-GGUF:Q4_K_XL",
            ],
        )
        mock_get.assert_called_once_with(
            "http://127.0.0.1:8083/v1/models", headers={}, timeout=3
        )

    def test_fetch_available_models_with_api_key(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "model-auth"}]}
        with patch.object(
            dashboard.requests, "get", return_value=mock_resp
        ) as mock_get:
            models = dashboard.fetch_available_models(
                "http://127.0.0.1:8083", api_key="sk-test-secret"
            )
        self.assertEqual(models, ["model-auth"])
        mock_get.assert_called_once_with(
            "http://127.0.0.1:8083/v1/models",
            headers={"Authorization": "Bearer sk-test-secret"},
            timeout=3,
        )

    def test_fetch_available_models_empty_data(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        with (
            patch.object(dashboard.requests, "get", return_value=mock_resp),
            self.assertRaisesRegex(ValueError, "No models found in response"),
        ):
            dashboard.fetch_available_models("http://127.0.0.1:8083")

    def test_fetch_available_models_404_response(self):
        import requests

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Client Error"
        )
        with (
            patch.object(dashboard.requests, "get", return_value=mock_resp),
            self.assertRaises(requests.exceptions.HTTPError),
        ):
            dashboard.fetch_available_models("http://127.0.0.1:8083")

    def test_fetch_available_models_500_response(self):
        import requests

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error"
        )
        with (
            patch.object(dashboard.requests, "get", return_value=mock_resp),
            self.assertRaises(requests.exceptions.HTTPError),
        ):
            dashboard.fetch_available_models("http://127.0.0.1:8083")

    def test_fetch_available_models_connection_timeout(self):
        import requests

        with (
            patch.object(
                dashboard.requests,
                "get",
                side_effect=requests.exceptions.Timeout("Connection timed out"),
            ),
            self.assertRaises(requests.exceptions.Timeout),
        ):
            dashboard.fetch_available_models("http://127.0.0.1:8083")

    def test_fetch_available_models_connection_error(self):
        import requests

        with (
            patch.object(
                dashboard.requests,
                "get",
                side_effect=requests.exceptions.ConnectionError("Connection refused"),
            ),
            self.assertRaises(requests.exceptions.ConnectionError),
        ):
            dashboard.fetch_available_models("http://127.0.0.1:8083")

    def test_fetch_available_models_ssrf_forbidden_ip(self):
        with self.assertRaisesRegex(ValueError, "Forbidden IP address range"):
            dashboard.fetch_available_models("http://169.254.169.254")

    def test_ui_model_retrieval_success_renders_selectbox(self):
        mock_fetch = MagicMock(return_value=["model-a", "model-b"])
        mock_streamlit = MagicMock()
        available_models = []
        try:
            available_models = mock_fetch("http://127.0.0.1:8083")
        except Exception as err:  # noqa: BLE001
            mock_streamlit.warning(
                f"Could not retrieve models from endpoint ({err}). You can enter a model identifier manually below."
            )

        if available_models:
            mock_streamlit.selectbox("Model ID / Endpoint Alias", available_models)
        else:
            mock_streamlit.text_input("Model ID / Endpoint Alias", value="Qwen3.6-27B")

        mock_streamlit.selectbox.assert_called_once_with(
            "Model ID / Endpoint Alias", ["model-a", "model-b"]
        )
        mock_streamlit.warning.assert_not_called()

    def test_ui_model_retrieval_failure_renders_warning_and_text_input(self):
        mock_fetch = MagicMock(side_effect=RuntimeError("Endpoint offline"))
        mock_streamlit = MagicMock()
        available_models = []
        try:
            available_models = mock_fetch("http://127.0.0.1:8083")
        except Exception as err:  # noqa: BLE001
            mock_streamlit.warning(
                f"Could not retrieve models from endpoint ({err}). You can enter a model identifier manually below."
            )

        if available_models:
            mock_streamlit.selectbox("Model ID / Endpoint Alias", available_models)
        else:
            mock_streamlit.text_input("Model ID / Endpoint Alias", value="Qwen3.6-27B")

        mock_streamlit.warning.assert_called_once_with(
            "Could not retrieve models from endpoint (Endpoint offline). You can enter a model identifier manually below."
        )
        mock_streamlit.text_input.assert_called_once_with(
            "Model ID / Endpoint Alias", value="Qwen3.6-27B"
        )
        mock_streamlit.selectbox.assert_not_called()

    def test_dashboard_execution_endpoint_offline(self):
        import importlib
        import requests

        with (
            patch.dict(
                sys.modules,
                {
                    "dashboard": dashboard,
                    "streamlit": mock_st,
                    "pandas": MagicMock(),
                    "plotly": MagicMock(),
                    "plotly.express": MagicMock(),
                    "plotly.graph_objects": MagicMock(),
                },
            ),
            patch(
                "requests.get",
                side_effect=requests.exceptions.ConnectionError("Offline"),
            ),
        ):
            importlib.reload(dashboard)


class TestEndpointsPersistence(unittest.TestCase):
    """Test endpoint persistence, JSON serialization, and CRUD helper operations."""

    def test_load_endpoints_default_creation_when_not_exists(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            endpoints_file = Path(tmp_dir) / "sub" / "endpoints.json"
            eps = dashboard.load_endpoints(endpoints_file)
            self.assertEqual(len(eps), 2)
            self.assertEqual(eps[0]["name"], "Local Llama Router")
            self.assertEqual(eps[0]["url"], "http://127.0.0.1:8083")
            self.assertTrue(eps[0]["is_default"])
            self.assertEqual(eps[1]["name"], "Production LLM-Routing")
            self.assertFalse(eps[1]["is_default"])
            self.assertTrue(endpoints_file.exists())

    def test_load_endpoints_loads_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            endpoints_file = Path(tmp_dir) / "endpoints.json"
            custom_data = [
                {
                    "name": "Custom Ep",
                    "url": "http://127.0.0.1:9999",
                    "api_key": "sk-custom",
                    "is_default": True,
                }
            ]
            with open(endpoints_file, "w", encoding="utf-8") as f:
                json.dump(custom_data, f)

            eps = dashboard.load_endpoints(endpoints_file)
            self.assertEqual(len(eps), 1)
            self.assertEqual(eps[0]["name"], "Custom Ep")
            self.assertEqual(eps[0]["api_key"], "sk-custom")

    def test_load_endpoints_corrupted_json_handling(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            endpoints_file = Path(tmp_dir) / "endpoints.json"
            with open(endpoints_file, "w", encoding="utf-8") as f:
                f.write("{invalid json: corrupt")

            eps = dashboard.load_endpoints(endpoints_file)
            self.assertEqual(len(eps), 2)
            self.assertEqual(eps[0]["name"], "Local Llama Router")

    def test_load_endpoints_empty_file_handling(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            endpoints_file = Path(tmp_dir) / "endpoints.json"
            with open(endpoints_file, "w", encoding="utf-8") as f:
                f.write("   \n")

            eps = dashboard.load_endpoints(endpoints_file)
            self.assertEqual(len(eps), 2)
            self.assertEqual(eps[0]["name"], "Local Llama Router")

    def test_load_endpoints_non_list_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            endpoints_file = Path(tmp_dir) / "endpoints.json"
            with open(endpoints_file, "w", encoding="utf-8") as f:
                json.dump({"error": "not a list"}, f)

            eps = dashboard.load_endpoints(endpoints_file)
            self.assertEqual(len(eps), 2)

    def test_load_endpoints_missing_fields_and_filtering(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            endpoints_file = Path(tmp_dir) / "endpoints.json"
            invalid_records = [
                "not a dict",
                {"url": "http://127.0.0.1:8080"},  # Missing name
                {"name": "No URL"},  # Missing url
                {"name": "   ", "url": "http://127.0.0.1:8080"},  # Blank name
                {"name": "Valid", "url": "http://127.0.0.1:8080"},  # Valid, defaults api_key and is_default
            ]
            with open(endpoints_file, "w", encoding="utf-8") as f:
                json.dump(invalid_records, f)

            eps = dashboard.load_endpoints(endpoints_file)
            self.assertEqual(len(eps), 1)
            self.assertEqual(eps[0]["name"], "Valid")
            self.assertEqual(eps[0]["api_key"], "")
            self.assertFalse(eps[0]["is_default"])

    def test_load_endpoints_all_corrupt_records_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            endpoints_file = Path(tmp_dir) / "endpoints.json"
            with open(endpoints_file, "w", encoding="utf-8") as f:
                json.dump([{"invalid": 1}, {"bad": 2}], f)

            eps = dashboard.load_endpoints(endpoints_file)
            self.assertEqual(len(eps), 2)

    def test_save_endpoints_atomic_write_and_persistence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            endpoints_file = Path(tmp_dir) / "nested" / "endpoints.json"
            data = [
                {
                    "name": "Router",
                    "url": "http://127.0.0.1:8083",
                    "api_key": "key123",
                    "is_default": True,
                }
            ]
            dashboard.save_endpoints(data, endpoints_file)
            self.assertTrue(endpoints_file.exists())
            self.assertFalse(endpoints_file.with_suffix(".tmp").exists())
            with open(endpoints_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["name"], "Router")

    def test_save_endpoints_validation_errors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            endpoints_file = Path(tmp_dir) / "endpoints.json"
            with self.assertRaisesRegex(ValueError, "Endpoints must be a list"):
                dashboard.save_endpoints("not-a-list", endpoints_file)
            with self.assertRaisesRegex(ValueError, "must be a dict"):
                dashboard.save_endpoints(["not-a-dict"], endpoints_file)
            with self.assertRaisesRegex(ValueError, "non-empty string 'name'"):
                dashboard.save_endpoints([{"url": "http://127.0.0.1:8080"}], endpoints_file)
            with self.assertRaisesRegex(ValueError, "non-empty string 'url'"):
                dashboard.save_endpoints([{"name": "Ep"}], endpoints_file)

    def test_add_endpoint_success_and_duplicate_handling(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            endpoints_file = Path(tmp_dir) / "endpoints.json"
            eps = dashboard.add_endpoint(
                "New Endpoint", "http://127.0.0.1:8090", "key-xyz", is_default=True, file_path=endpoints_file
            )
            self.assertEqual(len(eps), 3)
            new_ep = next(e for e in eps if e["name"] == "New Endpoint")
            self.assertTrue(new_ep["is_default"])
            # Old default should be unset
            old_def = next(e for e in eps if e["name"] == "Local Llama Router")
            self.assertFalse(old_def["is_default"])

            with self.assertRaisesRegex(ValueError, "already exists"):
                dashboard.add_endpoint(
                    "New Endpoint", "http://127.0.0.1:8091", file_path=endpoints_file
                )
            with self.assertRaisesRegex(ValueError, "Endpoint name cannot be empty"):
                dashboard.add_endpoint(
                    "", "http://127.0.0.1:8091", file_path=endpoints_file
                )

    def test_update_endpoint_success_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            endpoints_file = Path(tmp_dir) / "endpoints.json"
            dashboard.load_endpoints(endpoints_file)

            updated = dashboard.update_endpoint(
                "Local Llama Router",
                "Renamed Router",
                "http://127.0.0.1:8095",
                "new-key",
                is_default=True,
                file_path=endpoints_file,
            )
            names = [e["name"] for e in updated]
            self.assertIn("Renamed Router", names)
            self.assertNotIn("Local Llama Router", names)

            with self.assertRaisesRegex(ValueError, "not found"):
                dashboard.update_endpoint(
                    "NonExistent", "New", "http://127.0.0.1:8080", file_path=endpoints_file
                )
            with self.assertRaisesRegex(ValueError, "already exists"):
                dashboard.update_endpoint(
                    "Renamed Router", "Production LLM-Routing", "http://127.0.0.1:8080", file_path=endpoints_file
                )
            with self.assertRaisesRegex(ValueError, "Endpoint name cannot be empty"):
                dashboard.update_endpoint(
                    "Renamed Router", "", "http://127.0.0.1:8080", file_path=endpoints_file
                )

    def test_save_and_load_endpoints_non_string_api_key(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            endpoints_file = Path(tmp_dir) / "endpoints.json"
            dashboard.save_endpoints(
                [{"name": "Test", "url": "http://127.0.0.1:8080", "api_key": 9999}],
                endpoints_file,
            )
            eps = dashboard.load_endpoints(endpoints_file)
            self.assertEqual(eps[0]["api_key"], "")

            with open(endpoints_file, "w", encoding="utf-8") as f:
                json.dump([{"name": "Test2", "url": "http://127.0.0.1:8080", "api_key": 1234}], f)
            eps2 = dashboard.load_endpoints(endpoints_file)
            self.assertEqual(eps2[0]["api_key"], "")

    def test_delete_endpoint_success_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            endpoints_file = Path(tmp_dir) / "endpoints.json"
            eps = dashboard.load_endpoints(endpoints_file)
            self.assertEqual(len(eps), 2)

            with self.assertRaisesRegex(ValueError, "not found"):
                dashboard.delete_endpoint("NonExistent", file_path=endpoints_file)

            remaining = dashboard.delete_endpoint("Local Llama Router", file_path=endpoints_file)
            self.assertEqual(len(remaining), 1)
            self.assertEqual(remaining[0]["name"], "Production LLM-Routing")
            # Since default was deleted, remaining becomes default
            self.assertTrue(remaining[0]["is_default"])

            with self.assertRaisesRegex(ValueError, "Cannot delete the only"):
                dashboard.delete_endpoint("Production LLM-Routing", file_path=endpoints_file)


class TestEndpointManagementAndRunnerUI(unittest.TestCase):
    """Test UI endpoint management expander and runner configuration auth integration."""

    def test_fetch_available_models_strips_trailing_slash(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "model-1"}]}
        with patch.object(dashboard.requests, "get", return_value=mock_resp) as mock_get:
            models = dashboard.fetch_available_models("http://127.0.0.1:8083/")
            self.assertEqual(models, ["model-1"])
            mock_get.assert_called_once_with(
                "http://127.0.0.1:8083/v1/models", headers={}, timeout=3
            )

    def test_fetch_available_models_defensive_none_data_and_non_str_id(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                None,
                {"id": 12345},  # Non-string id ignored
                {"id": ""},  # Blank string ignored
                {"id": "valid-model"},
            ]
        }
        with patch.object(dashboard.requests, "get", return_value=mock_resp):
            models = dashboard.fetch_available_models("http://127.0.0.1:8083")
            self.assertEqual(models, ["valid-model"])

    def test_fetch_available_models_data_key_is_none(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": None}
        with (
            patch.object(dashboard.requests, "get", return_value=mock_resp),
            self.assertRaisesRegex(ValueError, "No models found in response"),
        ):
            dashboard.fetch_available_models("http://127.0.0.1:8083")

    def test_ui_manage_endpoints_save_action_validation(self):
        mock_st_local = MagicMock()
        ep_name = "   "
        if not ep_name.strip():
            mock_st_local.error("Endpoint name cannot be empty.")
        mock_st_local.error.assert_called_once_with("Endpoint name cannot be empty.")

    def test_ui_manage_endpoints_test_connection_success(self):
        mock_st_local = MagicMock()
        mock_fetch = MagicMock(return_value=["model-1", "model-2", "model-3"])
        with patch.object(dashboard, "fetch_available_models", mock_fetch):
            test_models = dashboard.fetch_available_models("http://127.0.0.1:8083", "secret")
            mock_st_local.success(
                f"Connection successful! {len(test_models)} models available: {', '.join(test_models[:5])}..."
            )
            mock_st_local.success.assert_called_once_with(
                "Connection successful! 3 models available: model-1, model-2, model-3..."
            )

    def test_ui_manage_endpoints_test_connection_failure(self):
        mock_st_local = MagicMock()
        err_msg = "Connection refused"
        mock_st_local.error(f"Connection failed: {err_msg}")
        mock_st_local.error.assert_called_once_with("Connection failed: Connection refused")

    def test_ui_manage_endpoints_save_action_success_and_error_paths(self):
        mock_st = MagicMock()
        mock_st.session_state = {}

        # 1. Add new endpoint success
        with patch.object(dashboard, "add_endpoint") as mock_add:
            valid_u = dashboard.validate_endpoint_url("http://127.0.0.1:8080", allow_private=True)
            mock_add("New", valid_u, "key", True)
            mock_st.session_state["endpoint_notice"] = "Endpoint 'New' saved successfully."
            mock_st.rerun()

            mock_add.assert_called_once_with("New", "http://127.0.0.1:8080", "key", True)
            self.assertEqual(mock_st.session_state["endpoint_notice"], "Endpoint 'New' saved successfully.")
            mock_st.rerun.assert_called_once()

        # 2. Update endpoint success
        mock_st.reset_mock()
        with patch.object(dashboard, "update_endpoint") as mock_update:
            valid_u = dashboard.validate_endpoint_url("http://127.0.0.1:8080", allow_private=True)
            mock_update("Old", "New", valid_u, "key", True)
            mock_st.session_state["endpoint_notice"] = "Endpoint 'New' updated successfully."
            mock_st.rerun()

            mock_update.assert_called_once_with("Old", "New", "http://127.0.0.1:8080", "key", True)
            self.assertEqual(mock_st.session_state["endpoint_notice"], "Endpoint 'New' updated successfully.")
            mock_st.rerun.assert_called_once()

        # 3. Invalid URL error
        mock_st.reset_mock()
        try:
            dashboard.validate_endpoint_url("invalid-url", allow_private=True)
        except ValueError as err:
            mock_st.error(f"Failed to save endpoint: {err}")
        mock_st.error.assert_called_once()

    def test_ui_manage_endpoints_delete_action_success_and_error(self):
        mock_st = MagicMock()
        mock_st.session_state = {}

        # Delete success
        with patch.object(dashboard, "delete_endpoint") as mock_delete:
            mock_delete("Target Ep")
            mock_st.session_state["endpoint_notice"] = "Endpoint 'Target Ep' deleted successfully."
            mock_st.rerun()

            mock_delete.assert_called_once_with("Target Ep")
            mock_st.rerun.assert_called_once()

        # Delete error
        mock_st.reset_mock()
        with patch.object(dashboard, "delete_endpoint", side_effect=ValueError("Cannot delete")):
            try:
                dashboard.delete_endpoint("Target Ep")
            except ValueError as err:
                mock_st.error(f"Failed to delete endpoint: {err}")
            mock_st.error.assert_called_once_with("Failed to delete endpoint: Cannot delete")

    def test_runner_cmd_appends_api_key_when_present(self):
        import sys

        valid_endpoint = "http://127.0.0.1:8083"
        valid_model = "Qwen3.6-27B"
        valid_corpus = "kld_corpus.txt"
        valid_tokens = 5000
        valid_gguf = "model.gguf"
        valid_api_key = "sk-runner-token"

        cmd = [
            sys.executable,
            "run_suite.py",
            "--mode",
            "all",
            "--endpoint",
            valid_endpoint,
            "--model",
            valid_model,
            "--tokens",
            str(valid_tokens),
            "--corpus",
            valid_corpus,
        ]
        if valid_gguf:
            cmd.extend(["--gguf-path", valid_gguf])
        if valid_api_key:
            cmd.extend(["--api-key", valid_api_key])

        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[1], "run_suite.py")
        self.assertIn("--api-key", cmd)
        key_idx = cmd.index("--api-key")
        self.assertEqual(cmd[key_idx + 1], "sk-runner-token")

    def test_runner_cmd_omits_api_key_when_empty(self):
        import sys

        valid_endpoint = "http://127.0.0.1:8083"
        valid_model = "Qwen3.6-27B"
        valid_corpus = "kld_corpus.txt"
        valid_tokens = 5000
        valid_gguf = ""
        valid_api_key = ""

        cmd = [
            sys.executable,
            "run_suite.py",
            "--mode",
            "all",
            "--endpoint",
            valid_endpoint,
            "--model",
            valid_model,
            "--tokens",
            str(valid_tokens),
            "--corpus",
            valid_corpus,
        ]
        if valid_gguf:
            cmd.extend(["--gguf-path", valid_gguf])
        if valid_api_key:
            cmd.extend(["--api-key", valid_api_key])

        self.assertNotIn("--api-key", cmd)
        self.assertNotIn("--gguf-path", cmd)


if __name__ == "__main__":
    unittest.main()
