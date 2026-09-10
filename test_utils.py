import pytest
from utils import validate_endpoint_url, validate_model_name

def test_validate_endpoint_url_valid():
    assert validate_endpoint_url("http://localhost:8080") == "http://localhost:8080"
    assert validate_endpoint_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"
    assert validate_endpoint_url("https://api.github.com") == "https://api.github.com"

def test_validate_endpoint_url_invalid_scheme():
    for scheme in ["ftp://example.com", "file:///etc/passwd", "gopher://example.com", "ws://example.com"]:
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            validate_endpoint_url(scheme)

def test_validate_endpoint_url_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_endpoint_url("")
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_endpoint_url(None)

def test_validate_endpoint_url_missing_hostname():
    for url in ["http://", "https://", "http://:8080", "http:///path"]:
        with pytest.raises(ValueError, match="missing hostname"):
            validate_endpoint_url(url)

def test_validate_endpoint_url_invalid_port():
    with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
        validate_endpoint_url("http://example.com:0")
    with pytest.raises(ValueError, match="Invalid URL port"):
        validate_endpoint_url("http://example.com:65536")
    with pytest.raises(ValueError, match="Invalid URL port"):
        validate_endpoint_url("http://example.com:-1")
    with pytest.raises(ValueError, match="Invalid URL port"):
        validate_endpoint_url("http://example.com:abc")

def test_validate_endpoint_url_unresolvable():
    from unittest.mock import patch
    import socket
    with patch("socket.getaddrinfo", side_effect=socket.gaierror("Name or service not known")):
        with pytest.raises(ValueError, match="Could not resolve hostname"):
            validate_endpoint_url("http://some-fake-host.example")

def test_validate_endpoint_url_forbidden_ip():
    with pytest.raises(ValueError, match="Forbidden IP address range"):
        validate_endpoint_url("http://10.0.0.1")
    with pytest.raises(ValueError, match="Forbidden IP address range"):
        validate_endpoint_url("http://192.168.0.1")

def test_validate_endpoint_url_multi_ip_with_private():
    from unittest.mock import patch
    import socket

    # Simulate hostname resolving to a public IP and a private IP
    mock_addrs = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('10.0.0.1', 0)),
    ]
    with patch("socket.getaddrinfo", return_value=mock_addrs):
        with pytest.raises(ValueError, match="Forbidden IP address range"):
            validate_endpoint_url("http://multi-ip.example.com")

def test_validate_model_name():
    assert validate_model_name("llama-3:8b") == "llama-3:8b"
    assert validate_model_name("Qwen/Qwen2.5-7B-Instruct") == "Qwen/Qwen2.5-7B-Instruct"
    assert validate_model_name("model_v1.0:latest-2026") == "model_v1.0:latest-2026"

    for invalid in ["invalid model name;", "model$(whoami)", "model|cat", "", None]:
        with pytest.raises(ValueError, match="Invalid model name"):
            validate_model_name(invalid)
