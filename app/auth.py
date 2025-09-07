# app/auth.py
from __future__ import annotations

from typing import Optional, Iterable
from flask import Request
import hmac

BEARER_PREFIX = "bearer "  # case-insensitive(büyük/küçük harf duyarsız)


def _normalize_set(values: Iterable[str] | None) -> set[str]:
    if not values:
        return set()
    return {v.strip() for v in values if isinstance(v, str) and v.strip()}


def _get_bearer_token(auth_header: str | None) -> Optional[str]:
    """RFC'ye yakın: 'Bearer <token>' biçimini boşluk sayısından bağımsız yakala."""
    if not auth_header:
        return None
    # case-insensitive kontrol
    if auth_header.lower().startswith(BEARER_PREFIX):
        parts = auth_header.split(" ", 1)
        if len(parts) == 2:
            return parts[1].strip() or None
    return None


def extract_token_from_headers(request: Request) -> Optional[str]:
    """
    Authorization: Bearer <token>  (önerilen)
    veya
    X-API-Key: <token>             (alternatif)
    """
    token = _get_bearer_token(request.headers.get("Authorization"))
    if token:
        return token
    alt = request.headers.get("X-API-Key")
    return alt.strip() if alt else None


def _contains_secure(token: str, allowed: set[str]) -> bool:
    # timing-attack'e dayanıklı karşılaştırma
    for t in allowed:
        if hmac.compare_digest(token, t):
            return True
    return False


def is_authorized(
    token: Optional[str],
    api_secret_tokens: Iterable[str] | set[str],
    api_keys: Iterable[str] | set[str],
) -> bool:
    if not token:
        return False
    secrets = _normalize_set(api_secret_tokens)
    keys = _normalize_set(api_keys)
    return _contains_secure(token, secrets) or _contains_secure(token, keys)
