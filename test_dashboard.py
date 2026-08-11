import unittest
import sys
from unittest.mock import MagicMock

# Simple MagicMock for dependencies works better because they support attribute access naturally.
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

# To avoid valid_gguf NameError, don't press the button!
mock_st.button.return_value = False

sys.modules['streamlit'] = mock_st
sys.modules['pandas'] = MagicMock()
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()

import dashboard

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
        with self.assertRaisesRegex(ValueError, "Endpoint URL cannot be empty."):
            dashboard.validate_endpoint_url("")
        with self.assertRaisesRegex(ValueError, "Endpoint URL cannot be empty."):
            dashboard.validate_endpoint_url(None)

    def test_invalid_scheme(self):
        with self.assertRaisesRegex(ValueError, "Invalid URL scheme 'ftp'. Only http and https are allowed."):
            dashboard.validate_endpoint_url("ftp://server.com")
        with self.assertRaisesRegex(ValueError, "Invalid URL scheme 'ws'. Only http and https are allowed."):
            dashboard.validate_endpoint_url("ws://localhost")

    def test_missing_hostname(self):
        with self.assertRaisesRegex(ValueError, "Invalid URL: missing hostname."):
            dashboard.validate_endpoint_url("http://")
        with self.assertRaisesRegex(ValueError, "Invalid URL: missing hostname."):
            dashboard.validate_endpoint_url("http:/path/only")

    def test_forbidden_ips(self):
        forbidden_ips = [
            "10.0.0.1", "172.16.0.1", "192.168.1.1", "169.254.169.254",
            "224.0.0.1", "240.0.0.1"
        ]
        # Skip IPv6 tests because `urllib.parse` parsing of bare IPv6 in `http://::1` might not yield `::1` as hostname if brackets are missing
        # e.g., urlparse('http://[::1]').hostname is '::1'
        # Let's test with properly formatted IPv6 URLs
        forbidden_ipv6 = ["[::1]", "[fe80::1]", "[fd00::1]"]

        for ip in forbidden_ips:
            with self.subTest(ip=ip):
                with self.assertRaisesRegex(ValueError, "Forbidden IP address range"):
                    dashboard.validate_endpoint_url(f"http://{ip}")

        for ip in forbidden_ipv6:
            with self.subTest(ip=ip):
                with self.assertRaisesRegex(ValueError, "Forbidden IP address range"):
                    dashboard.validate_endpoint_url(f"http://{ip}")

        with self.assertRaisesRegex(ValueError, "Forbidden IP address range"):
            dashboard.validate_endpoint_url("http://127.0.0.2")

if __name__ == '__main__':
    unittest.main()
