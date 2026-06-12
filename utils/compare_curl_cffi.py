#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare TLS fingerprint with and without curl_cffi impersonation.

Usage:
  python utils/compare_curl_cffi.py                 # compare fingerprints
  python utils/compare_curl_cffi.py --url https://example.com  # test specific URL
  python utils/compare_curl_cffi.py --impersonate safari15_5   # use different browser
"""

import asyncio
import json
import sys
import argparse

import httpx

FINGERPRINT_CHECK_URLS = [
    "https://tls.peet.ws/api/all",  # JA3/JA4 + HTTP/2 fingerprints
    "https://tools.scrapfly.io/api/fp/ja3",  # JA3 fingerprint + headers
]


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def test_standard_httpx(url: str):
    """Test with standard httpx transport (no impersonation)."""
    print_section("Standard httpx (no impersonation)")

    try:
        async with httpx.AsyncClient(http2=True, timeout=15) as client:
            resp = await client.get(url)
            print(f"Status: {resp.status_code}")
            print(f"HTTP Version: {resp.http_version}")
            _print_json_response(resp)
    except Exception as e:
        print(f"Error: {e}")


async def test_curl_cffi(url: str, impersonate: str):
    """Test with curl_cffi transport (browser TLS fingerprint impersonation)."""
    print_section(f"curl_cffi impersonating '{impersonate}'")

    try:
        from httpx_curl_cffi import AsyncCurlTransport, CurlOpt
    except ImportError:
        print("ERROR: httpx_curl_cffi is not installed. Run: pip install httpx_curl_cffi")
        return

    try:
        transport = AsyncCurlTransport(
            impersonate=impersonate,
            verify=True,
            curl_options={CurlOpt.FRESH_CONNECT: True},
        )
        async with httpx.AsyncClient(transport=transport, timeout=15) as client:
            resp = await client.get(url)
            print(f"Status: {resp.status_code}")
            print(f"HTTP Version: {resp.http_version}")
            _print_json_response(resp)
    except Exception as e:
        print(f"Error: {e}")


async def test_curl_cffi_no_impersonation(url: str):
    """Test with curl_cffi transport WITHOUT impersonation (raw curl fingerprint)."""
    print_section("curl_cffi transport (NO impersonation)")

    try:
        from httpx_curl_cffi import AsyncCurlTransport, CurlOpt
    except ImportError:
        print("ERROR: httpx_curl_cffi is not installed. Run: pip install httpx_curl_cffi")
        return

    try:
        transport = AsyncCurlTransport(
            verify=True,
            curl_options={CurlOpt.FRESH_CONNECT: True},
        )
        async with httpx.AsyncClient(transport=transport, timeout=15) as client:
            resp = await client.get(url)
            print(f"Status: {resp.status_code}")
            print(f"HTTP Version: {resp.http_version}")
            _print_json_response(resp)
    except Exception as e:
        print(f"Error: {e}")


def _print_json_response(resp: httpx.Response):
    """Pretty-print JSON response, focusing on fingerprint-related fields."""
    try:
        data = resp.json()
    except Exception:
        # Not JSON, print first 500 chars of text
        print(resp.text[:500])
        return

    # Extract key fingerprint fields for comparison
    _extract_and_print(data, "ja3", "JA3 Fingerprint")
    _extract_and_print(data, "ja3_hash", "JA3 Hash")
    _extract_and_print(data, "ja4", "JA4 Fingerprint")
    _extract_and_print(data, "akamai", "Akamai Fingerprint")
    _extract_and_print(data, "peetprint", "PeetPrint (HTTP/2)")
    _extract_and_print(data, "http_version", "HTTP Version")
    _extract_and_print(data, "tls", "TLS Info")
    _extract_and_print(data, "user_agent", "User-Agent")

    # Fallback: print the whole response
    if not any(k in str(data).lower() for k in ["ja3", "ja4", "tls"]):
        print(json.dumps(data, indent=2)[:2000])


def _extract_and_print(data, key: str, label: str):
    """Extract and print a specific key from the response data."""
    if isinstance(data, dict):
        if key in data:
            value = data[key]
            if isinstance(value, (dict, list)):
                print(f"\n{label}:")
                print(json.dumps(value, indent=2))
            else:
                print(f"{label}: {value}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and key in item:
                print(f"{label}: {item[key]}")


async def main():
    parser = argparse.ArgumentParser(description="Compare TLS fingerprints with and without curl_cffi impersonation")
    parser.add_argument(
        "--url",
        default=None,
        help="URL to test (default: use fingerprint-checking endpoints)",
    )
    parser.add_argument(
        "--impersonate",
        default="chrome",
        help="Browser fingerprint to impersonate (default: chrome)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all built-in fingerprint check URLs",
    )
    args = parser.parse_args()

    urls = [args.url] if args.url else FINGERPRINT_CHECK_URLS

    for url in urls:
        print(f"\n{'#'*60}")
        print(f"  Target: {url}")
        print(f"{'#'*60}")

        # Test 1: Standard httpx (no curl_cffi)
        await test_standard_httpx(url)

        # Test 2: curl_cffi without impersonation (raw curl fingerprint)
        await test_curl_cffi_no_impersonation(url)

        # Test 3: curl_cffi with browser impersonation
        await test_curl_cffi(url, args.impersonate)

        if not args.all:
            break

    print_section("Summary")
    print("""
Key comparison points:
  1. JA3/JA4 hash: should differ between standard httpx and curl_cffi
  2. curl_cffi with impersonation: JA3 should match a real browser
  3. HTTP/2 fingerprint (PeetPrint/Akamai): curl_cffi mimics real browser H2 settings

If JA3 hashes are different between the three tests, curl_cffi is working correctly.
If JA3 is the same between tests 1 & 2, curl_cffi's transport is active but not impersonating.
""")


if __name__ == "__main__":
    asyncio.run(main())
