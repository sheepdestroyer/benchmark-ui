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
mock_st.session_state = DictWithDefault({
    "new_endpoint": "http://localhost:8080",
    "new_model": "test-model",
    "new_api_key": "sk-1234",
    "endpoints": [],
    "suite_results": None,
    "matrix_results": None,
    "kld_results": None
})

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
    if "Endpoint URL" in label: return "http://localhost:8080"
    if "Model Name" in label: return "test-model"
    if "API Key" in label: return "sk-1234"
    if "Corpus Name" in label: return "kld_corpus.txt"
    if "GGUF Path" in label: return ""
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
    }
)
modules_patcher.start()
import dashboard
modules_patcher.stop()

class TestValidateEndpointUrl(unittest.TestCase):

    def test_valid_urls(self):
        self.assertEqual(dashboard.validate_endpoint_url("http://google.com"), "http://google.com")
        self.assertEqual(dashboard.validate_endpoint_url("https://api.github.com/v1"), "https://api.github.com/v1")
        self.assertEqual(dashboard.validate_endpoint_url("http://8.8.8.8"), "http://8.8.8.8")

    def test_localhost_exceptions(self):
        self.assertEqual(dashboard.validate_endpoint_url("http://localhost"), "http://localhost")
        self.assertEqual(dashboard.validate_endpoint_url("http://localhost:8080"), "http://localhost:8080")
        self.assertEqual(dashboard.validate_endpoint_url("http://127.0.0.1"), "http://127.0.0.1")
        self.assertEqual(dashboard.validate_endpoint_url("http://127.0.0.1:5000"), "http://127.0.0.1:5000")

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
            "10.0.0.1", "172.16.0.1", "192.168.1.1", "169.254.169.254",
            "224.0.0.1", "240.0.0.1"
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
            "12345"
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
            "model]"
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

    @patch('os.path.exists', return_value=False)
    def test_get_presets_config_missing_file(self, mock_exists):
        config = dashboard._get_presets_config()
        self.assertIsNone(config)

    @patch('os.path.exists', return_value=True)
    def test_get_presets_config_syntax_error(self, mock_exists):
        with patch('configparser.ConfigParser.read', side_effect=configparser.ParsingError('Invalid INI')):
            config = dashboard._get_presets_config()
            self.assertIsNone(config)

    @patch('os.path.exists', return_value=True)
    def test_get_presets_config_caching(self, mock_exists):
        sample_ini = """
[*]
flash-attn = true

[my-model]
alias = MyModel
hf-repo = org/my-model
parallel = 2
"""
        with patch('configparser.ConfigParser.read') as mock_read:
            def fake_read(filenames, encoding=None):
                # simulate successful read by populating sections
                return [filenames]
            mock_read.side_effect = fake_read
            
            c1 = dashboard._get_presets_config()
            c2 = dashboard._get_presets_config()
            self.assertIs(c1, c2)
            self.assertEqual(mock_read.call_count, 1)

    def test_map_repo_to_preset_alias_missing_config(self):
        with patch.object(dashboard, '_get_presets_config', return_value=None):
            self.assertEqual(dashboard.map_repo_to_preset_alias('unknown/model'), 'unknown/model')
            # Fallbacks should still work
            self.assertEqual(dashboard.map_repo_to_preset_alias('qwen3.6-27b-gguf:q4_k_s'), 'Qwen3.6-27B')
            self.assertEqual(dashboard.map_repo_to_preset_alias('qwen3.6-27b-mtp-gguf:q4_k_s'), 'Qwen3.6-27B-spec3')
            self.assertEqual(dashboard.map_repo_to_preset_alias('qwen3.6-35b-a3b-gguf:q4_k_s'), 'Qwen3.6-35B-A3B')
            self.assertEqual(dashboard.map_repo_to_preset_alias('gemma-4-test'), 'gemma4-26a4b-routing')
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
        with patch.object(dashboard, '_get_presets_config', return_value=cp):
            # Exact section match
            self.assertEqual(dashboard.map_repo_to_preset_alias('exact-model'), 'exact-model')
            # Substring alias match
            self.assertEqual(dashboard.map_repo_to_preset_alias('prefix/ExactModel-extra'), 'exact-model')
            # Repo match with mtp/spec condition
            self.assertEqual(dashboard.map_repo_to_preset_alias('org/some-repo-mtp'), 'mtp-spec-model')
            self.assertEqual(dashboard.map_repo_to_preset_alias('org/some-repo-standard'), 'base-model')
            # Unmatched returns input
            self.assertEqual(dashboard.map_repo_to_preset_alias('unknown-other'), 'unknown-other')

    def test_get_preset_metadata_missing_config(self):
        with patch.object(dashboard, '_get_presets_config', return_value=None):
            meta = dashboard.get_preset_metadata('any-model')
            self.assertEqual(meta['parallel'], '1')
            self.assertEqual(meta['flash_attn'], 'true')
            self.assertEqual(meta['spec_type'], 'None')

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
        with patch.object(dashboard, '_get_presets_config', return_value=cp):
            meta = dashboard.get_preset_metadata('custom-model')
            self.assertEqual(meta['flash_attn'], 'false')
            self.assertEqual(meta['global_param'], 'global_val')
            self.assertEqual(meta['parallel'], '4')
            self.assertEqual(meta['spec_type'], 'draft')
            # Default retained if not overridden
            self.assertEqual(meta['n_gpu_layers'], '99')


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
        self.assertEqual(dashboard.fmt_num(float('nan')), "N/A")

    def test_fmt_num_pd_na_eval(self):
        dashboard.pd.isna.return_value = True
        self.assertEqual(dashboard.fmt_num(123), "N/A")
        dashboard.pd.isna.return_value = False

    def test_fmt_num_array_ambiguity_protection(self):
        class DummyArray:
            def __bool__(self):
                raise ValueError("The truth value of an array with more than one element is ambiguous.")
        dashboard.pd.isna.return_value = DummyArray()
        self.assertEqual(dashboard.fmt_num([1, 2]), "[1, 2]")
        dashboard.pd.isna.return_value = False

    def test_fmt_num_booleans(self):
        self.assertEqual(dashboard.fmt_num(True), "True")
        self.assertEqual(dashboard.fmt_num(False), "False")

    def test_fmt_num_infinity(self):
        self.assertEqual(dashboard.fmt_num(float('inf')), "Inf")
        self.assertEqual(dashboard.fmt_num(-float('inf')), "-Inf")


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
        self.assertEqual(res[0], {"Model_Quant": "Model-A (q4_k_m)", "Test Suite": "Needle", "Score": 1.0})
        self.assertEqual(res[1], {"Model_Quant": "Model-A (q4_k_m)", "Test Suite": "RULER", "Score": 0.0})
        self.assertEqual(res[2], {"Model_Quant": "Model-B (q8_0)", "Test Suite": "Needle", "Score": 1.0})

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
            valid_run.write_text(json.dumps({
                "run_metadata": {"timestamp": "2026-01-01T00:00:00", "target_endpoint": "http://localhost:8080"},
                "model_settings": {"profile_alias": "test-model", "base_quantization": "Q4_K_S"},
                "throughput_metrics": {"prefill_speed": 120.0, "decode_speed": 35.0},
                "reasoning_accuracy": {"needle": "Pass"},
                "quantization_loss": {"perplexity": 5.4}
            }))

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
                    self.assertEqual(parsed_runs[0]["Filename"], "run_2026-01-01T00-00-00.json")
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
            run_file.write_text(json.dumps({
                "run_metadata": {
                    "timestamp": "2026-09-11T00:00:00",
                    "target_endpoint": "http://localhost:8080",
                    "cli_arguments": ["--tokens", "65536"]
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
                    "fit": "true"
                },
                "throughput_metrics": {"prefill_speed": 150.0, "decode_speed": 40.0, "ttft": 0.12},
                "reasoning_accuracy": {"needle": "Pass", "ruler": 0.95, "longbench": 0.88, "swe_bench": "N/A"},
                "quantization_loss": {"perplexity": 4.5, "mean_kld": 0.02, "same_top_match_percent": 98.5}
            }))

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


class TestDashboardValidators(unittest.TestCase):
    def test_validate_gguf_path_none_or_empty(self):
        self.assertIsNone(dashboard.validate_gguf_path(None))
        self.assertEqual(dashboard.validate_gguf_path(""), "")

    def test_validate_gguf_path_existing_allowed_file(self):
        # Tempfile inside cwd
        with tempfile.NamedTemporaryFile(dir=Path.cwd(), suffix=".gguf") as tmp:
            self.assertEqual(dashboard.validate_gguf_path(tmp.name), str(Path(tmp.name).resolve()))

        # Relative path inside cwd
        with tempfile.NamedTemporaryFile(dir=Path.cwd(), prefix="test_gguf_", suffix=".gguf") as tmp:
            rel_name = os.path.basename(tmp.name)
            self.assertEqual(dashboard.validate_gguf_path(rel_name), str(Path(tmp.name).resolve()))

        # Tempfile inside Path(__file__).parent
        with tempfile.NamedTemporaryFile(dir=Path(__file__).parent.resolve(), suffix=".gguf") as tmp:
            self.assertEqual(dashboard.validate_gguf_path(tmp.name), str(Path(tmp.name).resolve()))

        # Tempfiles inside patched Path.home() .cache and models directories
        with tempfile.TemporaryDirectory() as fake_home:
            cache_dir = Path(fake_home) / ".cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            models_dir = Path(fake_home) / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            with patch.object(Path, "home", return_value=Path(fake_home)):
                with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".gguf") as tmp_cache:
                    self.assertEqual(dashboard.validate_gguf_path(tmp_cache.name), str(Path(tmp_cache.name).resolve()))
                with tempfile.NamedTemporaryFile(dir=models_dir, suffix=".gguf") as tmp_models:
                    self.assertEqual(dashboard.validate_gguf_path(tmp_models.name), str(Path(tmp_models.name).resolve()))

    def test_validate_gguf_path_symlink(self):
        with tempfile.NamedTemporaryFile(dir=Path.cwd(), suffix=".gguf") as target:
            symlink = Path.cwd() / f"test_symlink_{os.path.basename(target.name)}"
            try:
                symlink.symlink_to(target.name)
                self.assertEqual(
                    dashboard.validate_gguf_path(str(symlink)),
                    str(Path(target.name).resolve())
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
                with patch.object(Path, "cwd", return_value=Path(mock_parent_dir)), \
                     patch.object(Path, "home", return_value=Path(mock_parent_dir)):
                    with self.assertRaisesRegex(ValueError, r"GGUF path escapes allowed parent directories"):
                        dashboard.validate_gguf_path(str(outside_file))

    def test_validate_corpus_name_valid(self):
        self.assertEqual(dashboard.validate_corpus_name("kld_corpus.txt"), "kld_corpus.txt")
        self.assertEqual(dashboard.validate_corpus_name("corpus.json"), "corpus.json")
        self.assertEqual(dashboard.validate_corpus_name("custom_eval"), "custom_eval")
        # Leading and trailing whitespace should be stripped
        self.assertEqual(dashboard.validate_corpus_name("  kld_corpus.txt  "), "kld_corpus.txt")

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
        self.assertEqual(dashboard.validate_corpus_name("/path/to/kld_corpus.txt"), "kld_corpus.txt")
        self.assertEqual(dashboard.validate_corpus_name("corpora/nested/dataset.csv"), "dataset.csv")
        self.assertEqual(dashboard.validate_corpus_name("./local/dir/test_corpus"), "test_corpus")

    def test_validate_gguf_path_rejects_non_gguf(self):
        # Non-.gguf files in cwd
        for ext in [".txt", ".py", ".bin", ".json", "", ".dat"]:
            with tempfile.NamedTemporaryFile(dir=Path.cwd(), suffix=ext) as tmp:
                with self.assertRaisesRegex(ValueError, r"GGUF file must have a \.gguf extension"):
                    dashboard.validate_gguf_path(tmp.name)

        # Case-insensitive: uppercase .GGUF is accepted
        with tempfile.NamedTemporaryFile(dir=Path.cwd(), suffix=".GGUF") as tmp:
            self.assertEqual(dashboard.validate_gguf_path(tmp.name), str(Path(tmp.name).resolve()))

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
                with self.assertRaisesRegex(ValueError, r"GGUF file must have a \.gguf extension"):
                    dashboard.validate_gguf_path(str(bashrc))

                # Rejects root_model.gguf in home root because it escapes allowed parent directories
                with self.assertRaisesRegex(ValueError, r"GGUF path escapes allowed parent directories"):
                    dashboard.validate_gguf_path(str(root_gguf))

                # Accepts .gguf in ~/.cache
                self.assertEqual(
                    dashboard.validate_gguf_path(str(cache_gguf)),
                    str(cache_gguf.resolve())
                )

                # Accepts .gguf in nested ~/.cache
                self.assertEqual(
                    dashboard.validate_gguf_path(str(nested_cache_gguf)),
                    str(nested_cache_gguf.resolve())
                )

                # Accepts .gguf in ~/models
                self.assertEqual(
                    dashboard.validate_gguf_path(str(models_gguf)),
                    str(models_gguf.resolve())
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
                with self.assertRaisesRegex(ValueError, r"Context length tokens must be between 1 and 262144\."):
                    dashboard.validate_new_tokens(out_val)

    def test_validate_new_tokens_invalid_type(self):
        for bad_val in [None, True, False, "abc", "", "12.34", 12.34, [5000], {"tokens": 5000}]:
            with self.subTest(val=bad_val):
                with self.assertRaisesRegex(ValueError, r"Context length tokens must be between 1 and 262144\."):
                    dashboard.validate_new_tokens(bad_val)


import importlib.util

HAS_PANDAS_AND_PLOTLY = (
    importlib.util.find_spec("pandas") is not None
    and importlib.util.find_spec("plotly") is not None
)


@unittest.skipUnless(HAS_PANDAS_AND_PLOTLY, "pandas and plotly required for throughput figure tests")
class TestBuildThroughputFigure(unittest.TestCase):
    def setUp(self):
        import plotly.graph_objects as real_go
        self._orig_go = dashboard.go
        dashboard.go = real_go

    def tearDown(self):
        dashboard.go = self._orig_go

    def test_empty_dataframe(self):
        import pandas as pd
        df = pd.DataFrame(columns=["Model", "KV Quant", "Context Length", "Prefill (t/s)", "Decode (t/s)"])
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
        self.assertEqual(len(dashboard.build_throughput_figure("invalid_string").data), 0)
        self.assertEqual(len(dashboard.build_throughput_figure([1, 2, 3]).data), 0)
        self.assertEqual(len(dashboard.build_throughput_figure(123).data), 0)

    def test_missing_and_nan_quants(self):
        import pandas as pd
        df = pd.DataFrame({
            "Model": ["M1", "M1"],
            "KV Quant": [None, float("nan")],
            "Context Length": [1024, 2048],
            "Prefill (t/s)": [100.0, 110.0],
            "Decode (t/s)": [30.0, 32.0]
        })
        fig = dashboard.build_throughput_figure(df)
        self.assertEqual(len(fig.data), 0)

    def test_multiple_models_and_quants_and_ordering(self):
        import pandas as pd
        df = pd.DataFrame({
            "Model": ["Model_B", "Model_A", "Model_A", "Model_A", "Model_A", "Model_Unknown"],
            "KV Quant": ["f16", "q8_0", "f16", "q5_1", "q4_0", "custom_quant"],
            "Context Length": [4096, 2048, 1024, 1024, 512, 128],
            "Prefill (t/s)": [100.0, 120.0, 150.0, 140.0, 160.0, 90.0],
            "Decode (t/s)": [30.0, 35.0, 45.0, 42.0, 48.0, 25.0]
        })
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
        df = pd.DataFrame({
            "Model": ["Model_A", "Model_A"],
            "KV Quant": ["f16", "f16"],
            "Context Length": [4096, 512],
            "Prefill (t/s)": [100.0, 150.0],
            "Decode (t/s)": [30.0, 45.0]
        })
        fig = dashboard.build_throughput_figure(df)
        t0 = fig.data[0]
        self.assertEqual(list(t0.x), [512, 4096])


if __name__ == "__main__":
    unittest.main()
