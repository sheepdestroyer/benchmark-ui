"""
Shared validation utilities for LLM benchmark UI and dashboard.
"""

import ipaddress
import math
import re
import socket
import urllib.parse

IPV4_RESERVED_240 = ipaddress.ip_network("240.0.0.0/4")


def validate_endpoint_url(url_str, allow_private: bool = False):
    if not url_str or not isinstance(url_str, str) or not url_str.strip():
        raise ValueError("Endpoint URL cannot be empty.")
    url_str = url_str.strip()
    parsed = urllib.parse.urlparse(url_str)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Invalid URL scheme '{parsed.scheme}'. Only http and https are allowed."
        )
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid URL: missing hostname.")

    try:
        port = parsed.port
    except ValueError as e:
        raise ValueError(f"Invalid port: {e}")
    if port is not None and (port < 1 or port > 65535):
        raise ValueError(
            f"Invalid port number: {port}. Port must be between 1 and 65535."
        )

    hostname_lower = hostname.lower()
    if hostname_lower == "localhost" or hostname == "127.0.0.1":
        return url_str

    try:
        addrs = socket.getaddrinfo(hostname, None, family=socket.AF_UNSPEC)
        for addr in addrs:
            ip_str = addr[4][0]
            ip = ipaddress.ip_address(ip_str)
            if allow_private:
                is_forbidden = (
                    ip.is_link_local
                    or ip.is_multicast
                    or ip.is_unspecified
                    or (ip.is_reserved and not ip.is_loopback)
                    or (ip.version == 4 and ip in IPV4_RESERVED_240)
                )
                if is_forbidden:
                    raise ValueError(
                        f"Forbidden IP address range: {hostname} ({ip_str})"
                    )
            else:
                if (
                    ip.is_private
                    or ip.is_loopback
                    or ip.is_link_local
                    or ip.is_multicast
                    or ip.is_reserved
                    or not ip.is_global
                ):
                    raise ValueError(
                        f"Forbidden IP address range: {hostname} ({ip_str})"
                    )
    except (socket.gaierror, socket.herror):
        raise ValueError(f"Could not resolve hostname: {hostname}")
    except ValueError as e:
        if "Forbidden IP address range" in str(e):
            raise
    return url_str


def validate_model_name(model_name):
    if not model_name or not re.match(r"^[a-zA-Z0-9._:/-]+$", model_name):
        raise ValueError(
            f"Invalid model name '{model_name}'. Must match ^[a-zA-Z0-9._:/-]+$"
        )
    return model_name


def redact_cli_args(args: list[str] | None) -> list[str]:
    """Redact sensitive arguments (such as API keys) from CLI argument lists."""
    if not args:
        return []
    redacted = []
    skip_next = False
    for arg in args:
        if skip_next:
            redacted.append("********")
            skip_next = False
        elif arg == "--api-key":
            redacted.append(arg)
            skip_next = True
        elif arg.startswith("--api-key="):
            redacted.append("--api-key=********")
        else:
            redacted.append(arg)
    if skip_next:
        redacted.append("********")
    return redacted


CONTEXT_TIERS = {
    "8k": 8192,
    "32k": 32768,
    "64k": 65536,
    "128k": 131072,
    "240k": 240000,
}


def parse_context_tokens(tokens_val):
    """Parse context token length from integer or tier string (e.g., '8k', '32k', '240k')."""
    if isinstance(tokens_val, bool):
        raise TypeError(
            f"tokens must be an integer, float, or string, got {type(tokens_val).__name__}"
        )

    if isinstance(tokens_val, int):
        val = tokens_val
    elif isinstance(tokens_val, float):
        if math.isinf(tokens_val) or math.isnan(tokens_val):
            raise ValueError(f"Invalid context tokens: '{tokens_val}'")
        if not tokens_val.is_integer():
            raise ValueError(
                f"Context tokens must be an integer value, got non-integer float: {tokens_val}"
            )
        val = int(tokens_val)
    elif isinstance(tokens_val, str):
        cleaned = tokens_val.strip().lower()
        if not cleaned:
            raise ValueError("Context tokens cannot be empty string")
        if cleaned in CONTEXT_TIERS:
            val = CONTEXT_TIERS[cleaned]
        elif cleaned.endswith("k"):
            prefix = cleaned[:-1]
            try:
                num = float(prefix)
            except ValueError:
                raise ValueError(
                    f"Invalid context tokens or tier specification: '{tokens_val}'"
                )
            if math.isinf(num) or math.isnan(num):
                raise ValueError(
                    f"Invalid context tokens or tier specification: '{tokens_val}'"
                )
            token_float = num * 1024
            if not token_float.is_integer():
                raise ValueError(
                    f"Context tokens must resolve to an integer value, got: {token_float}"
                )
            val = int(token_float)
        else:
            try:
                num = float(cleaned)
            except ValueError:
                raise ValueError(
                    f"Invalid context tokens or tier specification: '{tokens_val}'"
                )
            if math.isinf(num) or math.isnan(num):
                raise ValueError(
                    f"Invalid context tokens or tier specification: '{tokens_val}'"
                )
            if not num.is_integer():
                raise ValueError(
                    f"Context tokens must be an integer value, got non-integer float: {tokens_val}"
                )
            val = int(num)
    else:
        raise TypeError(
            f"tokens must be an integer, float, or string, got {type(tokens_val).__name__}"
        )

    if val <= 0:
        raise ValueError(f"Context tokens must be positive (> 0), got {val}")

    return val
