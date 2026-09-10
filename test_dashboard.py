import configparser
import unittest
import sys
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


class TestSafeCacheDataAndLoadRuns(unittest.TestCase):
    def test_safe_cache_data_fallback_with_mock(self):
        # When st.cache_data is a mock (as in test env), safe_cache_data returns original func
        def dummy_func():
            return 42

        decorated = dashboard.safe_cache_data(ttl=60)(dummy_func)
        self.assertIs(decorated, dummy_func)
        self.assertEqual(decorated(), 42)

    def test_safe_cache_data_with_real_cache_data(self):
        # Simulate real st.cache_data function
        called_with = {}

        def fake_cache_data(**kwargs):
            called_with.update(kwargs)
            def decorator(f):
                f.__cached__ = True
                return f
            return decorator

        with patch.object(dashboard.st, "cache_data", fake_cache_data):
            def dummy_func():
                return 99

            decorated = dashboard.safe_cache_data(ttl=60)(dummy_func)
            self.assertEqual(called_with, {"ttl": 60})
            self.assertTrue(getattr(decorated, "__cached__", False))
            self.assertEqual(decorated(), 99)

    def test_load_runs_returns_dataframe(self):
        # load_runs should return a pandas DataFrame (or mocked DF object) without raising exception
        df = dashboard.load_runs()
        self.assertIsNotNone(df)


if __name__ == "__main__":
    unittest.main()
