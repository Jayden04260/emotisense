"""
Spotify user login (Authorization Code + PKCE) - EmotiSense "My Spotify"
features.

This is deliberately separate from the Client Credentials flow used
elsewhere in app/app.py for album-art/embed lookups. Client Credentials
is app-only auth and can never touch a specific person's account; the 4
features this module supports all act on the *logged-in user's own*
Spotify account, which needs a real login:

  1. Cross-referencing their recently-played tracks against EmotiSense's
     prediction history.
  2. Auto-creating a real playlist in their account from that history.
  3. Playing/queueing a recommended track on one of their active devices
     (requires Spotify Premium - Spotify's Web API playback-control
     endpoints reject free accounts).
  4. Saving a recommended track to their Liked Songs.

Kept Streamlit-free and dependency-light (just `requests`) so it can be
imported and unit-tested on its own; app/app.py wires this into
`st.query_params` for the redirect leg and `st.session_state`/a small
local JSON file for token storage.

IMPORTANT - Spotify's Feb 2025 security update (mandatory for all apps
since Nov 2025):
  - Redirect URIs must use the loopback IP literal "127.0.0.1", not the
    hostname "localhost" - Spotify now rejects "localhost" outright.
  - PKCE is required for every client type, including confidential
    (server-side, client-secret-holding) apps like this one - so this
    module always does Authorization Code *with* PKCE, never plain
    Authorization Code.

Every network-calling function here follows the same convention as the
existing Client Credentials code in app.py: best-effort, returns
None/[]/False (or a (False, reason) tuple where the failure reason is
worth surfacing) rather than raising, so a flaky connection or an
expired token degrades gracefully instead of crashing the app.
"""

import base64
import hashlib
import json
import re
import secrets
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE = "https://api.spotify.com/v1"

# Scopes needed for the 4 login-gated features:
#  - user-read-private            resolve the user's own Spotify ID
#                                  (needed to create a playlist "owned by"
#                                  them) and display name
#  - user-read-recently-played    recently-played cross-reference
#  - playlist-modify-private/
#    playlist-modify-public       auto-playlist creation
#  - user-modify-playback-state,
#    user-read-playback-state     direct playback control / queueing
#  - user-library-modify          save to Liked Songs
SCOPES = [
    "user-read-private",
    "user-read-recently-played",
    "playlist-modify-private",
    "playlist-modify-public",
    "user-modify-playback-state",
    "user-read-playback-state",
    "user-library-modify",
]


# --------------------------------------------------------------------------
# PKCE + authorize URL
# --------------------------------------------------------------------------

def generate_pkce_pair():
    """Returns (code_verifier, code_challenge) per RFC 7636: a random
    43-128 character string, and its SHA-256 hash base64url-encoded with
    no padding."""
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


def generate_state():
    """A random, unguessable value to protect against CSRF on the OAuth
    redirect - the caller stores this and checks it matches what Spotify
    sends back."""
    return secrets.token_urlsafe(16)


def build_authorize_url(client_id, redirect_uri, code_challenge, state, scopes=None):
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "code_challenge_method": "S256",
        "code_challenge": code_challenge,
        "state": state,
        "scope": " ".join(scopes or SCOPES),
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


# --------------------------------------------------------------------------
# Token exchange / refresh
# --------------------------------------------------------------------------

def _tokens_from_response(data, fallback_refresh_token=None):
    return {
        "access_token": data.get("access_token"),
        "refresh_token": data.get("refresh_token") or fallback_refresh_token,
        # 60s safety margin so a token isn't treated as valid for a call
        # that then takes just long enough to actually expire mid-flight.
        "expires_at": time.time() + float(data.get("expires_in", 3600)) - 60,
        "scope": data.get("scope", ""),
    }


def exchange_code_for_tokens(client_id, client_secret, redirect_uri, code, code_verifier, timeout=8):
    """Swaps an authorization code for an access + refresh token. Returns
    a dict (access_token/refresh_token/expires_at/scope) on success, or
    None on any failure - bad/expired/reused code, wrong redirect URI,
    wrong code_verifier, network error, etc."""
    try:
        response = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "code_verifier": code_verifier,
            },
            auth=(client_id, client_secret),
            timeout=timeout,
        )
        response.raise_for_status()
        return _tokens_from_response(response.json())
    except Exception:
        return None


def refresh_tokens(client_id, client_secret, refresh_token, timeout=8):
    """Uses a refresh token to get a fresh access token. Spotify doesn't
    always rotate the refresh token itself, so this falls back to
    reusing the old one when a new one isn't returned. Returns None if
    the refresh itself fails - typically because the user revoked
    EmotiSense's access from their Spotify account settings - in which
    case the caller should treat them as logged out."""
    try:
        response = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
            },
            auth=(client_id, client_secret),
            timeout=timeout,
        )
        response.raise_for_status()
        return _tokens_from_response(response.json(), fallback_refresh_token=refresh_token)
    except Exception:
        return None


def is_expired(tokens):
    return not tokens or not tokens.get("access_token") or time.time() >= tokens.get("expires_at", 0)


# --------------------------------------------------------------------------
# Local token persistence (a small gitignored JSON file, not st.cache_*,
# since it must survive across Streamlit reruns *and* full app restarts -
# re-logging in every time you restart the app would defeat the point).
# --------------------------------------------------------------------------

def load_tokens(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def save_tokens(path, tokens):
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tokens), encoding="utf-8")
    except Exception:
        pass


def clear_tokens(path):
    path = Path(path)
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass


# --------------------------------------------------------------------------
# API calls that need the user's own access token (as opposed to the
# Client Credentials app token used elsewhere in app.py)
# --------------------------------------------------------------------------

def _auth_headers(access_token):
    return {"Authorization": f"Bearer {access_token}"}


def get_current_user(access_token, timeout=8):
    """Returns {"id": ..., "display_name": ...} for the logged-in user,
    or None on failure. The id is required (not just cosmetic) - creating
    a playlist needs it."""
    try:
        r = requests.get(f"{API_BASE}/me", headers=_auth_headers(access_token), timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return {"id": data.get("id"), "display_name": data.get("display_name") or data.get("id")}
    except Exception:
        return None


def get_recently_played(access_token, limit=50, timeout=8):
    """Returns a list of {name, artist, played_at, uri} dicts, most
    recent first, or [] on any failure (including "no history yet")."""
    try:
        r = requests.get(
            f"{API_BASE}/me/player/recently-played",
            headers=_auth_headers(access_token),
            params={"limit": limit},
            timeout=timeout,
        )
        r.raise_for_status()
        items = r.json().get("items", [])
        return [
            {
                "name": item["track"]["name"],
                "artist": ", ".join(a["name"] for a in item["track"].get("artists", [])),
                "played_at": item.get("played_at", ""),
                "uri": item["track"]["uri"],
            }
            for item in items
        ]
    except Exception:
        return []


def strip_featured_artists(text: str) -> str:
    """Strips a trailing/parenthetical "feat. X" ("featuring X" / "ft. X")
    credit from a song or artist string before querying Spotify's search
    API. data/music.csv credits songs the way they're commonly written
    ("D12 feat. Eminem", "Bounce Out With That Remix (feat. Machine Gun
    Kelly)"), but Spotify's own artist field for the track only lists the
    primary artist - querying with the raw credited string reliably fails
    to match even though the track exists under the shorter name."""
    text = re.sub(r"\s*\(feat\.?.*?\)\s*$", "", text, flags=re.IGNORECASE).strip()
    text = re.split(r"\s+(?:feat\.|featuring|ft\.)\s+", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return text


def search_track(access_token, song, artist, timeout=8):
    """Resolves a song/artist from the curated CSV to a real Spotify
    track, using the *user's* access token rather than the separate
    Client Credentials app token - so playback/save/playlist actions work
    even for someone who's only configured login, not the app token, and
    vice versa. Returns {"id": ..., "uri": ...} or None if not found."""
    song, artist = strip_featured_artists(song), strip_featured_artists(artist)
    try:
        r = requests.get(
            f"{API_BASE}/search",
            headers=_auth_headers(access_token),
            params={"q": f"track:{song} artist:{artist}", "type": "track", "limit": 1},
            timeout=timeout,
        )
        r.raise_for_status()
        items = r.json().get("tracks", {}).get("items", [])
        if not items:
            return None
        track = items[0]
        return {"id": track["id"], "uri": track["uri"]}
    except Exception:
        return None


def create_playlist(access_token, user_id, name, description="", public=False, timeout=8):
    """Creates an empty playlist owned by user_id. Returns the new
    playlist's id, or None on failure."""
    try:
        r = requests.post(
            f"{API_BASE}/users/{user_id}/playlists",
            headers={**_auth_headers(access_token), "Content-Type": "application/json"},
            json={"name": name, "description": description, "public": public},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("id")
    except Exception:
        return None


def add_tracks_to_playlist(access_token, playlist_id, track_uris, timeout=8):
    """Adds track URIs to a playlist. Spotify caps this endpoint at 100
    URIs per call; EmotiSense's curated dataset is far smaller than that,
    so no chunking is implemented here - callers with more than 100
    should chunk themselves. Returns True/False."""
    try:
        r = requests.post(
            f"{API_BASE}/playlists/{playlist_id}/tracks",
            headers={**_auth_headers(access_token), "Content-Type": "application/json"},
            json={"uris": track_uris[:100]},
            timeout=timeout,
        )
        r.raise_for_status()
        return True
    except Exception:
        return False


def play_track(access_token, track_uri, device_id=None, timeout=8):
    """Starts playback of a single track on the given device, or
    whichever device is currently active if device_id is None. Requires
    Spotify Premium *and* an already-open/active Spotify app somewhere -
    both are common failure modes worth surfacing distinctly rather than
    failing silently like the read-only calls above, so this returns
    (True, None) on success or (False, reason) on failure."""
    try:
        params = {"device_id": device_id} if device_id else {}
        r = requests.put(
            f"{API_BASE}/me/player/play",
            headers={**_auth_headers(access_token), "Content-Type": "application/json"},
            params=params,
            json={"uris": [track_uri]},
            timeout=timeout,
        )
        if r.status_code == 204:
            return True, None
        if r.status_code == 403:
            return False, "Playback control needs Spotify Premium."
        if r.status_code == 404:
            return False, "No active Spotify device found - open Spotify on your phone or desktop first."
        r.raise_for_status()
        return True, None
    except Exception as e:
        return False, str(e)


def queue_track(access_token, track_uri, device_id=None, timeout=8):
    """Adds a track to the end of the user's current playback queue.
    Same Premium/active-device requirements and failure modes as
    play_track above."""
    try:
        params = {"uri": track_uri}
        if device_id:
            params["device_id"] = device_id
        r = requests.post(
            f"{API_BASE}/me/player/queue",
            headers=_auth_headers(access_token),
            params=params,
            timeout=timeout,
        )
        if r.status_code == 204:
            return True, None
        if r.status_code == 403:
            return False, "Queueing needs Spotify Premium."
        if r.status_code == 404:
            return False, "No active Spotify device found - open Spotify on your phone or desktop first."
        r.raise_for_status()
        return True, None
    except Exception as e:
        return False, str(e)


def save_track(access_token, track_id, timeout=8):
    """Adds a track to the user's Liked Songs. Returns True/False."""
    try:
        r = requests.put(
            f"{API_BASE}/me/tracks",
            headers={**_auth_headers(access_token), "Content-Type": "application/json"},
            json={"ids": [track_id]},
            timeout=timeout,
        )
        return r.status_code == 200
    except Exception:
        return False
