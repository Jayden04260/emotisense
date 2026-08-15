"""
Automated test suite for src/spotify_oauth.py - the Authorization Code +
PKCE login backing EmotiSense's "My Spotify" tab (see README "Spotify
Login (Optional)").

Covers everything that doesn't need a real network call: PKCE generation
(RFC 7636 shape), authorize URL construction, token-expiry math, and disk
persistence. The functions that actually call Spotify's API
(exchange_code_for_tokens, refresh_tokens, get_current_user, etc.) are
deliberately not mocked here - they're thin try/except wrappers around a
single requests call each, and their real behavior (does Spotify actually
reject a bad code the way we assume) can only be honestly checked against
Spotify's real endpoints, not a guess at what they return.

Run from the project root with:

    pytest
"""

import base64
import hashlib
import re
import time

import pytest

import spotify_oauth as sp


# --------------------------------------------------------------------------
# PKCE generation (RFC 7636)
# --------------------------------------------------------------------------

def test_generate_pkce_pair_verifier_length_is_in_rfc_range():
    verifier, _ = sp.generate_pkce_pair()
    # RFC 7636 requires 43-128 characters.
    assert 43 <= len(verifier) <= 128


def test_generate_pkce_pair_verifier_is_url_safe_charset():
    verifier, _ = sp.generate_pkce_pair()
    # RFC 7636 unreserved charset: [A-Za-z0-9-._~]
    assert re.fullmatch(r"[A-Za-z0-9\-._~]+", verifier)


def test_generate_pkce_pair_challenge_is_correct_sha256_of_verifier():
    verifier, challenge = sp.generate_pkce_pair()
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")
    assert challenge == expected


def test_generate_pkce_pair_is_random_each_call():
    v1, c1 = sp.generate_pkce_pair()
    v2, c2 = sp.generate_pkce_pair()
    assert v1 != v2
    assert c1 != c2


def test_generate_state_is_random_and_nonempty():
    s1 = sp.generate_state()
    s2 = sp.generate_state()
    assert s1 and s2
    assert s1 != s2


# --------------------------------------------------------------------------
# Authorize URL construction
# --------------------------------------------------------------------------

def test_build_authorize_url_contains_required_pkce_params():
    url = sp.build_authorize_url("client123", "http://127.0.0.1:8501", "chal", "state123")
    assert url.startswith(sp.AUTHORIZE_URL)
    assert "client_id=client123" in url
    assert "code_challenge=chal" in url
    assert "code_challenge_method=S256" in url
    assert "state=state123" in url
    assert "response_type=code" in url


def test_build_authorize_url_uses_loopback_ip_not_localhost():
    # Spotify's Feb 2025 security update rejects the "localhost" hostname
    # outright - a regression here would silently break every login.
    url = sp.build_authorize_url("client123", "http://127.0.0.1:8501", "chal", "state123")
    assert "127.0.0.1" in url
    assert "localhost" not in url


def test_build_authorize_url_default_scopes_cover_all_four_features():
    url = sp.build_authorize_url("client123", "http://127.0.0.1:8501", "chal", "state123")
    for scope in sp.SCOPES:
        assert scope in url


def test_build_authorize_url_respects_custom_scopes():
    url = sp.build_authorize_url(
        "client123", "http://127.0.0.1:8501", "chal", "state123", scopes=["user-read-private"]
    )
    assert "user-read-private" in url
    assert "playlist-modify-public" not in url


# --------------------------------------------------------------------------
# Token expiry math
# --------------------------------------------------------------------------

def test_tokens_from_response_applies_60s_safety_margin():
    data = {"access_token": "abc", "refresh_token": "def", "expires_in": 3600, "scope": "x"}
    before = time.time()
    tokens = sp._tokens_from_response(data)
    after = time.time()
    # expires_at should be ~3540s out (3600 - 60), not the raw 3600.
    assert before + 3540 - 1 <= tokens["expires_at"] <= after + 3540 + 1


def test_tokens_from_response_falls_back_to_old_refresh_token_when_absent():
    data = {"access_token": "new_access", "expires_in": 3600, "scope": "x"}
    tokens = sp._tokens_from_response(data, fallback_refresh_token="old_refresh")
    assert tokens["refresh_token"] == "old_refresh"


def test_tokens_from_response_prefers_new_refresh_token_when_present():
    data = {"access_token": "new_access", "refresh_token": "new_refresh", "expires_in": 3600, "scope": "x"}
    tokens = sp._tokens_from_response(data, fallback_refresh_token="old_refresh")
    assert tokens["refresh_token"] == "new_refresh"


def test_is_expired_true_for_none():
    assert sp.is_expired(None) is True


def test_is_expired_true_for_missing_access_token():
    assert sp.is_expired({"expires_at": time.time() + 3600}) is True


def test_is_expired_true_when_past_expiry():
    assert sp.is_expired({"access_token": "x", "expires_at": time.time() - 1}) is True


def test_is_expired_false_when_comfortably_valid():
    assert sp.is_expired({"access_token": "x", "expires_at": time.time() + 3600}) is False


# --------------------------------------------------------------------------
# Disk persistence
# --------------------------------------------------------------------------

def test_save_then_load_tokens_round_trips(tmp_path):
    path = tmp_path / "tokens.json"
    tokens = {"access_token": "a", "refresh_token": "b", "expires_at": 123.0, "scope": "x"}
    sp.save_tokens(str(path), tokens)
    assert sp.load_tokens(str(path)) == tokens


def test_load_tokens_returns_none_when_file_missing(tmp_path):
    path = tmp_path / "does_not_exist.json"
    assert sp.load_tokens(str(path)) is None


def test_load_tokens_returns_none_on_corrupt_json(tmp_path):
    path = tmp_path / "tokens.json"
    path.write_text("not valid json{{{", encoding="utf-8")
    assert sp.load_tokens(str(path)) is None


def test_clear_tokens_removes_the_file(tmp_path):
    path = tmp_path / "tokens.json"
    sp.save_tokens(str(path), {"access_token": "a"})
    assert path.exists()
    sp.clear_tokens(str(path))
    assert not path.exists()


def test_clear_tokens_is_a_noop_when_file_already_missing(tmp_path):
    path = tmp_path / "does_not_exist.json"
    sp.clear_tokens(str(path))  # must not raise
    assert not path.exists()


def test_save_tokens_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "tokens.json"
    sp.save_tokens(str(path), {"access_token": "a"})
    assert path.exists()


# --------------------------------------------------------------------------
# strip_featured_artists (search-query normalisation - data/music.csv
# credits songs the way they're commonly written, e.g. "D12 feat.
# Eminem", but Spotify's own artist field only lists the primary artist,
# so searching with the raw credited string reliably fails to match)
# --------------------------------------------------------------------------

def test_strip_featured_artists_strips_trailing_feat_from_artist():
    assert sp.strip_featured_artists("D12 feat. Eminem") == "D12"


def test_strip_featured_artists_strips_parenthetical_feat_from_song():
    assert (
        sp.strip_featured_artists("Bounce Out With That Remix (feat. Machine Gun Kelly)")
        == "Bounce Out With That Remix"
    )


def test_strip_featured_artists_handles_featuring_and_ft_variants():
    assert sp.strip_featured_artists("Lil Wayne featuring Drake") == "Lil Wayne"
    assert sp.strip_featured_artists("Lil Wayne ft. Drake") == "Lil Wayne"


def test_strip_featured_artists_leaves_plain_strings_untouched():
    assert sp.strip_featured_artists("Sister Sledge") == "Sister Sledge"
    assert sp.strip_featured_artists("Lost in Music") == "Lost in Music"
