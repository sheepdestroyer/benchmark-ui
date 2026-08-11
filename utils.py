"""
Shared validation utilities for LLM benchmark UI and dashboard.
"""

import urllib.parse
import ipaddress
import socket
import re

def validate_endpoint_url(url_str):
    if not url_str:
        raise ValueError("Endpoint URL cannot be empty.")
    parsed = urllib.parse.urlparse(url_str)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError(f"Invalid URL scheme '{parsed.scheme}'. Only http and https are allowed.")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid URL: missing hostname.")
    
    hostname_lower = hostname.lower()
    if hostname_lower == "localhost" or hostname == "127.0.0.1":
        return url_str
    
    try:
        addrs = socket.getaddrinfo(hostname, None, family=socket.AF_UNSPEC)
        for addr in addrs:
            ip_str = addr[4][0]
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or not ip.is_global:
                raise ValueError(f"Forbidden IP address range: {hostname} ({ip_str})")
    except socket.gaierror:
        raise ValueError(f"Could not resolve hostname: {hostname}")
    except ValueError as e:
        if "Forbidden IP address range" in str(e):
            raise
        pass
    return url_str

def validate_model_name(model_name):
    if not model_name or not re.match(r'^[a-zA-Z0-9._:/-]+$', model_name):
        raise ValueError(f"Invalid model name '{model_name}'. Must match ^[a-zA-Z0-9._:/-]+$")
    return model_name
