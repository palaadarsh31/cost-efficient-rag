"""Check whether this machine can reach Anthropic without using or reading an API key.

Exit codes: 0 = DNS, TCP/TLS and HTTP connection succeeded; 2 = DNS failed;
3 = TCP/TLS failed; 4 = HTTP connection failed without a server response.
This does NOT check API-key validity, account billing, model access, or quotas.
"""
from __future__ import annotations

import socket
import ssl
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HOST = "api.anthropic.com"
URL = "https://api.anthropic.com/v1/messages"


def main() -> int:
    print(f"Checking access from THIS computer to {HOST} (no API key used).")
    try:
        addresses = socket.getaddrinfo(HOST, 443, type=socket.SOCK_STREAM)
        if not addresses:
            raise socket.gaierror("no addresses returned")
    except socket.gaierror as exc:
        print(f"DNS check FAILED: {exc}")
        print("Check internet/DNS, VPN, firewall, or corporate proxy settings.")
        return 2
    print("DNS check passed.")

    try:
        with socket.create_connection((HOST, 443), timeout=12) as sock:
            with ssl.create_default_context().wrap_socket(sock, server_hostname=HOST):
                pass
    except (OSError, ssl.SSLError) as exc:
        print(f"TCP/TLS check FAILED: {type(exc).__name__}: {exc}")
        print("Your network may be blocking HTTPS to Anthropic.")
        return 3
    print("TCP/TLS check passed.")

    # Deliberately perform an unauthenticated GET. 401/403/404/405 are useful:
    # the remote HTTP server was reached; they do not prove key validity.
    request = Request(URL, method="GET", headers={"User-Agent": "problem2-network-check/1.0"})
    try:
        with urlopen(request, timeout=12) as response:
            status = response.status
    except HTTPError as exc:
        status = exc.code
    except (URLError, TimeoutError, OSError) as exc:
        print(f"HTTP check FAILED: {type(exc).__name__}: {exc}")
        return 4

    print(f"Anthropic server responded with HTTP {status} (no credentials sent).")
    print("Network access works. This check does NOT validate your API key or model access.")
    print("Use a freshly rotated key through the judgepipe --prompt-for-key option for a live test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
