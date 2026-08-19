import pytest
from url_validation import validate_endpoint_url

def test_validate_endpoint_url_valid():
    assert validate_endpoint_url("http://localhost:8080") == "http://localhost:8080"
    assert validate_endpoint_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"
    assert validate_endpoint_url("https://api.github.com") == "https://api.github.com"
    # test ipv6 localhost? maybe not needed, but wait it checks hostname
    # assert validate_endpoint_url("http://[::1]:8080") == "http://[::1]:8080"

def test_validate_endpoint_url_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_endpoint_url("")

def test_validate_endpoint_url_invalid_chars():
    with pytest.raises(ValueError, match="URL contains invalid characters"):
        validate_endpoint_url("http://example.com/api?val=1 space")
    with pytest.raises(ValueError, match="URL contains invalid characters"):
        validate_endpoint_url("http://example.com; rm -rf /")

def test_validate_endpoint_url_invalid_scheme():
    with pytest.raises(ValueError, match="Invalid URL scheme"):
        validate_endpoint_url("ftp://example.com")

def test_validate_endpoint_url_missing_hostname():
    with pytest.raises(ValueError, match="missing hostname"):
        validate_endpoint_url("http://")

def test_validate_endpoint_url_forbidden_ip():
    with pytest.raises(ValueError, match="Forbidden IP address range"):
        validate_endpoint_url("http://192.168.1.1:8080")
    with pytest.raises(ValueError, match="Forbidden IP address range"):
        validate_endpoint_url("http://10.0.0.1")
    with pytest.raises(ValueError, match="Forbidden IP address range"):
        validate_endpoint_url("http://172.16.0.1")
    with pytest.raises(ValueError, match="Forbidden IP address range"):
        validate_endpoint_url("http://169.254.169.254")
