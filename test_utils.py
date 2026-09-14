import re
import socket
from unittest.mock import patch

import pytest

from utils import (
    CONTEXT_TIERS,
    parse_context_tokens,
    redact_cli_args,
    validate_endpoint_url,
    validate_model_name,
)


def test_validate_endpoint_url_valid():
    assert validate_endpoint_url("http://localhost:8080") == "http://localhost:8080"
    assert validate_endpoint_url("http://localhost:1") == "http://localhost:1"
    assert validate_endpoint_url("http://localhost:65535") == "http://localhost:65535"
    assert validate_endpoint_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"
    assert validate_endpoint_url("https://api.github.com") == "https://api.github.com"
    assert validate_endpoint_url("  http://localhost:8080  ") == "http://localhost:8080"


def test_validate_endpoint_url_empty_and_whitespace():
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_endpoint_url("")
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_endpoint_url(None)
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_endpoint_url("   ")
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_endpoint_url("\t\n  ")
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_endpoint_url(12345)


def test_validate_endpoint_url_missing_or_invalid_scheme():
    # Missing scheme
    with pytest.raises(ValueError, match="Invalid URL scheme"):
        validate_endpoint_url("example.com")
    with pytest.raises(ValueError, match="Invalid URL scheme"):
        validate_endpoint_url("//example.com")
    with pytest.raises(ValueError, match="Invalid URL scheme"):
        validate_endpoint_url("://example.com")

    # Unsupported schemes
    unsupported_schemes = [
        "ftp://example.com",
        "file:///etc/passwd",
        "gopher://example.com",
        "ssh://example.com",
        "ws://example.com",
        "wss://example.com",
        "ldap://example.com",
    ]
    for url in unsupported_schemes:
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            validate_endpoint_url(url)


def test_validate_endpoint_url_missing_hostname():
    missing_hostname_urls = [
        "http://",
        "https://",
        "http://:8080",
        "http:///path/only",
    ]
    for url in missing_hostname_urls:
        with pytest.raises(ValueError, match="missing hostname"):
            validate_endpoint_url(url)


def test_validate_endpoint_url_invalid_ports():
    # Non-numeric ports
    for url in [
        "http://example.com:abc",
        "http://localhost:abc",
        "https://example.com:80a",
    ]:
        with pytest.raises(ValueError, match="Invalid port"):
            validate_endpoint_url(url)

    # Ports < 1
    for url in ["http://example.com:0", "http://localhost:0"]:
        with pytest.raises(ValueError, match="Invalid port number"):
            validate_endpoint_url(url)
    with pytest.raises(ValueError, match="Invalid port"):
        validate_endpoint_url("http://example.com:-1")

    # Ports > 65535
    for url in [
        "http://example.com:65536",
        "http://localhost:65536",
        "http://example.com:99999",
    ]:
        with pytest.raises(ValueError, match="Invalid port"):
            validate_endpoint_url(url)


def test_validate_endpoint_url_dns_resolution_failures():
    # Mock socket.gaierror
    with patch(
        "socket.getaddrinfo",
        side_effect=socket.gaierror(socket.EAI_NONAME, "Name or service not known"),
    ):
        with pytest.raises(
            ValueError, match="Could not resolve hostname: mock-gaierror.test"
        ):
            validate_endpoint_url("http://mock-gaierror.test")

    # Mock socket.herror
    with patch("socket.getaddrinfo", side_effect=socket.herror(1, "Unknown host")):
        with pytest.raises(
            ValueError, match="Could not resolve hostname: mock-herror.test"
        ):
            validate_endpoint_url("http://mock-herror.test")

    # Unresolvable domain in practice
    with pytest.raises(ValueError, match="Could not resolve hostname"):
        validate_endpoint_url("http://nonexistent-host-that-does-not-exist.invalid")


def test_validate_endpoint_url_multi_ip_resolution_blocking():
    # Multi-IP where second IP is private 10.0.0.1
    addrs_private = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 80)),
    ]
    with patch("socket.getaddrinfo", return_value=addrs_private):
        with pytest.raises(
            ValueError,
            match=r"Forbidden IP address range: multi-ip\.test \(10\.0\.0\.1\)",
        ):
            validate_endpoint_url("http://multi-ip.test")

    # Multi-IP where second IP is loopback 127.0.0.1
    addrs_loopback = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80)),
    ]
    with patch("socket.getaddrinfo", return_value=addrs_loopback):
        with pytest.raises(
            ValueError,
            match=r"Forbidden IP address range: multi-ip\.test \(127\.0\.0\.1\)",
        ):
            validate_endpoint_url("http://multi-ip.test")

    # Multi-IP where first IP is private
    addrs_priv_first = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.100", 80)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
    ]
    with patch("socket.getaddrinfo", return_value=addrs_priv_first):
        with pytest.raises(ValueError, match="Forbidden IP address range"):
            validate_endpoint_url("http://multi-ip.test")

    # Multi-IP where second IP is IPv6 loopback ::1
    addrs_ipv6_loopback = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 80, 0, 0)),
    ]
    with patch("socket.getaddrinfo", return_value=addrs_ipv6_loopback):
        with pytest.raises(ValueError, match="Forbidden IP address range"):
            validate_endpoint_url("http://multi-ip.test")

    # Multi-IP where all IPs are public (must succeed)
    addrs_public = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.35", 80)),
    ]
    with patch("socket.getaddrinfo", return_value=addrs_public):
        assert (
            validate_endpoint_url("http://multi-public.example.com")
            == "http://multi-public.example.com"
        )


def test_validate_endpoint_url_ipv6_loopback_and_private():
    # IPv6 loopback
    with pytest.raises(ValueError, match="Forbidden IP address range: ::1"):
        validate_endpoint_url("http://[::1]:8080")
    with pytest.raises(ValueError, match="Forbidden IP address range: ::1"):
        validate_endpoint_url("http://[::1]")
    with pytest.raises(ValueError, match="Forbidden IP address range: 0:0:0:0:0:0:0:1"):
        validate_endpoint_url("https://[0:0:0:0:0:0:0:1]:443")

    # IPv6 link-local
    with pytest.raises(ValueError, match="Forbidden IP address range: fe80::1"):
        validate_endpoint_url("http://[fe80::1]:8080")

    # IPv6 unique-local (private)
    with pytest.raises(ValueError, match="Forbidden IP address range: fc00::1"):
        validate_endpoint_url("http://[fc00::1]:8080")
    with pytest.raises(
        ValueError, match=r"Forbidden IP address range: fd12:3456:789a::1"
    ):
        validate_endpoint_url("http://[fd12:3456:789a::1]:8080")

    # IPv6 public global address (must succeed)
    addrs_ipv6_global = [
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2606:4700:4700::1111", 80, 0, 0))
    ]
    with patch("socket.getaddrinfo", return_value=addrs_ipv6_global):
        assert (
            validate_endpoint_url("http://[2606:4700:4700::1111]:80")
            == "http://[2606:4700:4700::1111]:80"
        )


def test_validate_endpoint_url_forbidden_ip():
    with pytest.raises(ValueError, match="Forbidden IP address range"):
        validate_endpoint_url("http://10.0.0.1")
    with pytest.raises(ValueError, match="Forbidden IP address range"):
        validate_endpoint_url("http://192.168.0.1")


def test_validate_endpoint_url_allow_private_allowed_ips():
    private_urls = [
        "http://10.0.0.1:8080",
        "http://172.16.0.1:8080",
        "http://192.168.0.1:8080",
        "http://127.0.0.2:8080",
        "http://[::1]:8080",
        "http://[fd00::1]:8080",
        "http://localhost:8083",
        "http://127.0.0.1:8083",
    ]
    for url in private_urls:
        assert validate_endpoint_url(url, allow_private=True) == url


def test_validate_endpoint_url_allow_private_forbidden_ips():
    forbidden_urls = [
        ("http://169.254.169.254", "169.254.169.254"),
        ("http://169.254.169.254:80", "169.254.169.254"),
        ("http://[fe80::1]:8080", "fe80::1"),
        ("http://224.0.0.1:8080", "224.0.0.1"),
        ("http://240.0.0.1:8080", "240.0.0.1"),
        ("http://0.0.0.0:8080", "0.0.0.0"),
        ("http://[::]:8080", "::"),
    ]
    for url, host in forbidden_urls:
        with pytest.raises(
            ValueError, match=f"Forbidden IP address range: {re.escape(host)}"
        ):
            validate_endpoint_url(url, allow_private=True)


def test_validate_endpoint_url_allow_private_dns_resolution():
    # Domain resolving to link-local is rejected under allow_private=True
    addrs_link_local = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 80))
    ]
    with (
        patch("socket.getaddrinfo", return_value=addrs_link_local),
        pytest.raises(
            ValueError,
            match=r"Forbidden IP address range: metadata\.local \(169\.254\.169\.254\)",
        ),
    ):
        validate_endpoint_url("http://metadata.local", allow_private=True)

    # Domain resolving to private IP is allowed under allow_private=True
    addrs_private = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.0.35", 80))]
    with patch("socket.getaddrinfo", return_value=addrs_private):
        assert (
            validate_endpoint_url(
                "http://llama-router.vendeuvre.lan", allow_private=True
            )
            == "http://llama-router.vendeuvre.lan"
        )


def test_validate_endpoint_url_default_rejects_private():
    private_urls = [
        "http://10.0.0.1:8080",
        "http://172.16.0.1:8080",
        "http://192.168.0.1:8080",
        "http://127.0.0.2:8080",
        "http://[::1]:8080",
        "http://[fd00::1]:8080",
    ]
    for url in private_urls:
        # Default allow_private=False
        with pytest.raises(ValueError, match="Forbidden IP address range"):
            validate_endpoint_url(url)
        # Explicit allow_private=False
        with pytest.raises(ValueError, match="Forbidden IP address range"):
            validate_endpoint_url(url, allow_private=False)


def test_validate_endpoint_url_non_ip_addrinfo_fallback():
    # If getaddrinfo returns an addr tuple that cannot be parsed as an IP address,
    # it catches ValueError and continues.
    addrs_invalid_ip = [(socket.AF_UNSPEC, 0, 0, "", ("non-ip-address-string", 0))]
    with patch("socket.getaddrinfo", return_value=addrs_invalid_ip):
        assert (
            validate_endpoint_url("http://custom-service.domain.test")
            == "http://custom-service.domain.test"
        )


def test_validate_model_name():
    assert validate_model_name("llama-3:8b") == "llama-3:8b"
    assert (
        validate_model_name("meta-llama/Llama-2-7b-chat-hf")
        == "meta-llama/Llama-2-7b-chat-hf"
    )
    assert validate_model_name("qwen2.5-coder:32b") == "qwen2.5-coder:32b"
    assert validate_model_name("model_v1.0") == "model_v1.0"

    for invalid in [
        "",
        None,
        "invalid model name;",
        "model rm -rf",
        "model$name",
        "model\nname",
    ]:
        with pytest.raises(ValueError, match="Invalid model name"):
            validate_model_name(invalid)


def test_redact_cli_args():
    assert redact_cli_args(None) == []
    assert redact_cli_args([]) == []
    assert redact_cli_args(["--mode", "all", "--model", "Qwen"]) == [
        "--mode",
        "all",
        "--model",
        "Qwen",
    ]
    assert redact_cli_args(["--api-key", "secret123", "--mode", "all"]) == [
        "--api-key",
        "********",
        "--mode",
        "all",
    ]
    assert redact_cli_args(["--api-key=secret123", "--mode", "all"]) == [
        "--api-key=********",
        "--mode",
        "all",
    ]
    assert redact_cli_args(["--mode", "all", "--api-key"]) == [
        "--mode",
        "all",
        "--api-key",
        "********",
    ]


def test_context_tiers_constant():
    assert CONTEXT_TIERS == {
        "8k": 8192,
        "32k": 32768,
        "64k": 65536,
        "128k": 131072,
        "240k": 240000,
    }


def test_parse_context_tokens_valid_tiers():
    assert parse_context_tokens("8k") == 8192
    assert parse_context_tokens("32k") == 32768
    assert parse_context_tokens("64k") == 65536
    assert parse_context_tokens("128k") == 131072
    assert parse_context_tokens("240k") == 240000
    # Case insensitivity and whitespace stripping
    assert parse_context_tokens("  8K  ") == 8192
    assert parse_context_tokens("\t32k\n") == 32768
    assert parse_context_tokens("240K") == 240000


def test_parse_context_tokens_arbitrary_k():
    assert parse_context_tokens("16k") == 16384
    assert parse_context_tokens("0.5k") == 512
    assert parse_context_tokens("1.25k") == 1280
    assert parse_context_tokens("+8k") == 8192


def test_parse_context_tokens_numeric_integers():
    assert parse_context_tokens(8192) == 8192
    assert parse_context_tokens(1) == 1
    assert parse_context_tokens("8192") == 8192
    assert parse_context_tokens("  32768  ") == 32768
    assert parse_context_tokens(8192.0) == 8192
    assert parse_context_tokens("8192.0") == 8192


def test_parse_context_tokens_reject_booleans():
    with pytest.raises(TypeError, match="tokens must be an integer, float, or string"):
        parse_context_tokens(True)
    with pytest.raises(TypeError, match="tokens must be an integer, float, or string"):
        parse_context_tokens(False)


def test_parse_context_tokens_reject_non_integer_floats():
    with pytest.raises(ValueError, match="must be an integer value"):
        parse_context_tokens(8192.5)
    with pytest.raises(ValueError, match="must be an integer value"):
        parse_context_tokens("8192.5")
    with pytest.raises(ValueError, match="must resolve to an integer value"):
        parse_context_tokens("0.333k")


def test_parse_context_tokens_reject_non_positive():
    for zero_or_neg in (0, -1, -50, -8192, 0.0, -0.0, -50.0):
        with pytest.raises(ValueError, match="must be positive"):
            parse_context_tokens(zero_or_neg)
    for zero_or_neg_str in (
        "0",
        "-0",
        "-50",
        "-8192",
        "0.0",
        "-0.0",
        "-50.0",
        "-8k",
        "0k",
        "-0k",
    ):
        with pytest.raises(ValueError, match="must be positive"):
            parse_context_tokens(zero_or_neg_str)


def test_parse_context_tokens_reject_infinities_and_nans():
    for inf_or_nan in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="Invalid context tokens"):
            parse_context_tokens(inf_or_nan)
    for inf_or_nan_str in ("inf", "-inf", "nan", "+inf", "infk", "-infk", "nank"):
        with pytest.raises(ValueError, match="Invalid context tokens"):
            parse_context_tokens(inf_or_nan_str)


def test_parse_context_tokens_reject_invalid_strings():
    for invalid in ("", "   ", "\t\n", "invalid", "abc", "k", "-k", "8kk", "none"):
        with pytest.raises(ValueError, match="Invalid context tokens|cannot be empty"):
            parse_context_tokens(invalid)


def test_parse_context_tokens_reject_invalid_types():
    for bad_type in (None, [8192], {"tokens": 8192}, (8192,), object()):
        with pytest.raises(
            TypeError, match="tokens must be an integer, float, or string"
        ):
            parse_context_tokens(bad_type)
