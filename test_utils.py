import pytest
from utils import validate_endpoint_url, validate_model_name

def test_validate_endpoint_url_valid():
    assert validate_endpoint_url("http://localhost:8080") == "http://localhost:8080"
    assert validate_endpoint_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"
    assert validate_endpoint_url("https://api.github.com") == "https://api.github.com"

def test_validate_endpoint_url_invalid_scheme():
    with pytest.raises(ValueError, match="Invalid URL scheme"):
        validate_endpoint_url("ftp://example.com")

def test_validate_endpoint_url_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_endpoint_url("")

def test_validate_model_name():
    assert validate_model_name("llama-3:8b") == "llama-3:8b"
    with pytest.raises(ValueError):
        validate_model_name("invalid model name;")

def test_validate_endpoint_url_missing_hostname():
    with pytest.raises(ValueError, match="missing hostname"):
        validate_endpoint_url("http://")

def test_validate_endpoint_url_unresolvable():
    with pytest.raises(ValueError, match="Could not resolve hostname"):
        validate_endpoint_url("http://nonexistent-host-that-does-not-exist.invalid")

def test_validate_endpoint_url_forbidden_ip():
    with pytest.raises(ValueError, match="Forbidden IP address range"):
        validate_endpoint_url("http://10.0.0.1")
    with pytest.raises(ValueError, match="Forbidden IP address range"):
        validate_endpoint_url("http://192.168.0.1")
