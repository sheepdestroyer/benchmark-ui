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
