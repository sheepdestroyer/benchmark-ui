"""
Shared validation utilities for LLM benchmark UI and dashboard.
"""

import ipaddress
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
