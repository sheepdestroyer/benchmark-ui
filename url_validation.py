import re
import urllib.parse
import ipaddress

# Security check: strict regex to prevent command/argument injection
ENDPOINT_URL_SAFE_CHARS_RE = re.compile(r'^[a-zA-Z0-9\-\._~:/\?#\[\]@!\*\+,=%]+$')

def validate_endpoint_url(url_str):
    if not url_str:
        raise ValueError("Endpoint URL cannot be empty.")

    if not ENDPOINT_URL_SAFE_CHARS_RE.match(url_str):
        raise ValueError("URL contains invalid characters")

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
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or not ip.is_global:
            raise ValueError(f"Forbidden IP address range: {hostname}")
    except ValueError as e:
        if "Forbidden IP address range" in str(e):
            raise
        pass
    
    return url_str
