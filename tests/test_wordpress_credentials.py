"""
Tests for WordPressCredentials URL validation and CredentialStore loading.

Invariants:
  - WordPressCredentials accepts only HTTPS URLs.
  - HTTP, FTP, file://, bare hostnames, localhost, and empty strings are rejected.
  - Rejection happens at model construction time — all creation paths are covered.
  - CredentialStore.load() surfaces the file path in CredentialInvalidError when
    the JSON contains a non-HTTPS URL.
  - Valid HTTPS credentials continue to load and save without change.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from services.credential_store import (
    CredentialInvalidError,
    CredentialStore,
    WordPressCredentials,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_creds(**overrides) -> dict:
    base = {
        "url": "https://example.com",
        "user": "wp-admin",
        "app_password": "xxxx xxxx xxxx xxxx xxxx xxxx",
    }
    return {**base, **overrides}


def _make_store_with_creds(client_id: str, website_id: str, payload: dict) -> tuple[CredentialStore, Path]:
    """Write a credential JSON to a temp dir and return (store, path)."""
    tmp = Path(tempfile.mkdtemp())
    cred_dir = tmp / client_id
    cred_dir.mkdir(parents=True)
    cred_file = cred_dir / f"{website_id}.json"
    cred_file.write_text(json.dumps(payload), encoding="utf-8")
    return CredentialStore(tmp), cred_file


# ── WordPressCredentials — valid HTTPS ────────────────────────────────────────

class TestWordPressCredentialsValidHttps:
    def test_https_url_accepted(self):
        creds = WordPressCredentials(**_valid_creds())
        assert creds.url == "https://example.com"

    def test_https_with_path_accepted(self):
        creds = WordPressCredentials(**_valid_creds(url="https://example.com/wp"))
        assert creds.url == "https://example.com/wp"

    def test_https_with_subdomain_accepted(self):
        creds = WordPressCredentials(**_valid_creds(url="https://www.overheaddoordenver.com"))
        assert creds.url == "https://www.overheaddoordenver.com"

    def test_https_with_port_accepted(self):
        creds = WordPressCredentials(**_valid_creds(url="https://staging.example.com:8443"))
        assert creds.url == "https://staging.example.com:8443"

    def test_other_fields_unaffected_by_url_validator(self):
        creds = WordPressCredentials(**_valid_creds())
        assert creds.user == "wp-admin"
        assert creds.app_password == "xxxx xxxx xxxx xxxx xxxx xxxx"
        assert creds.default_category_id is None

    def test_default_category_id_accepted_with_valid_url(self):
        creds = WordPressCredentials(**_valid_creds(default_category_id=5))
        assert creds.default_category_id == 5


# ── WordPressCredentials — invalid URLs rejected ──────────────────────────────

class TestWordPressCredentialsInvalidUrl:
    def _assert_rejected(self, url: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            WordPressCredentials(**_valid_creds(url=url))
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("url",) for e in errors), (
            f"Expected a 'url' field error for {url!r}, got: {errors}"
        )

    def test_http_rejected(self):
        self._assert_rejected("http://example.com")

    def test_http_www_rejected(self):
        self._assert_rejected("http://www.overheaddoordenver.com")

    def test_ftp_rejected(self):
        self._assert_rejected("ftp://example.com")

    def test_file_scheme_rejected(self):
        self._assert_rejected("file:///etc/passwd")

    def test_empty_string_rejected(self):
        self._assert_rejected("")

    def test_bare_hostname_rejected(self):
        self._assert_rejected("example.com")

    def test_localhost_no_scheme_rejected(self):
        self._assert_rejected("localhost")

    def test_localhost_http_rejected(self):
        self._assert_rejected("http://localhost")

    def test_localhost_https_accepted(self):
        # localhost over HTTPS is valid (used in local dev with self-signed certs)
        creds = WordPressCredentials(**_valid_creds(url="https://localhost"))
        assert creds.url == "https://localhost"

    def test_https_no_netloc_rejected(self):
        # "https://" alone — scheme present but no hostname
        self._assert_rejected("https://")

    def test_missing_scheme_with_slashes_rejected(self):
        self._assert_rejected("//example.com")

    def test_error_message_contains_got_value(self):
        with pytest.raises(ValidationError) as exc_info:
            WordPressCredentials(**_valid_creds(url="http://example.com"))
        msg = str(exc_info.value)
        assert "http://example.com" in msg

    def test_error_message_mentions_https(self):
        with pytest.raises(ValidationError) as exc_info:
            WordPressCredentials(**_valid_creds(url="http://example.com"))
        msg = str(exc_info.value)
        assert "HTTPS" in msg


# ── CredentialStore.load() — HTTP URL in file ─────────────────────────────────

class TestCredentialStoreHttpsEnforcement:
    def test_valid_https_credential_loads(self):
        store, _ = _make_store_with_creds("RIMC", "overheaddoordenver", _valid_creds())
        creds = store.load("RIMC", "overheaddoordenver")
        assert creds.url == "https://example.com"

    def test_http_url_in_file_raises_credential_invalid_error(self):
        store, _ = _make_store_with_creds(
            "RIMC", "overheaddoordenver",
            _valid_creds(url="http://overheaddoordenver.com"),
        )
        with pytest.raises(CredentialInvalidError) as exc_info:
            store.load("RIMC", "overheaddoordenver")
        msg = str(exc_info.value)
        assert "overheaddoordenver" in msg   # site identity present

    def test_http_error_message_contains_file_path(self):
        store, cred_file = _make_store_with_creds(
            "RIMC", "overheaddoordenver",
            _valid_creds(url="http://overheaddoordenver.com"),
        )
        with pytest.raises(CredentialInvalidError) as exc_info:
            store.load("RIMC", "overheaddoordenver")
        msg = str(exc_info.value)
        # CredentialStore.load() includes the path in the CredentialInvalidError prefix
        assert "overheaddoordenver.json" in msg

    def test_http_error_message_mentions_url_field(self):
        store, _ = _make_store_with_creds(
            "RIMC", "overheaddoordenver",
            _valid_creds(url="http://overheaddoordenver.com"),
        )
        with pytest.raises(CredentialInvalidError) as exc_info:
            store.load("RIMC", "overheaddoordenver")
        msg = str(exc_info.value)
        assert "url" in msg.lower()

    def test_ftp_url_in_file_raises_credential_invalid_error(self):
        store, _ = _make_store_with_creds(
            "RIMC", "overheaddoordenver",
            _valid_creds(url="ftp://overheaddoordenver.com"),
        )
        with pytest.raises(CredentialInvalidError):
            store.load("RIMC", "overheaddoordenver")

    def test_bare_hostname_in_file_raises_credential_invalid_error(self):
        store, _ = _make_store_with_creds(
            "RIMC", "overheaddoordenver",
            _valid_creds(url="overheaddoordenver.com"),
        )
        with pytest.raises(CredentialInvalidError):
            store.load("RIMC", "overheaddoordenver")

    def test_empty_url_in_file_raises_credential_invalid_error(self):
        store, _ = _make_store_with_creds(
            "RIMC", "overheaddoordenver",
            _valid_creds(url=""),
        )
        with pytest.raises(CredentialInvalidError):
            store.load("RIMC", "overheaddoordenver")


# ── Direct construction paths (setup-site, import-sites) ─────────────────────

class TestDirectConstruction:
    """
    WordPressCredentials is also constructed directly in main.py (setup-site,
    import-sites commands). The field validator covers these paths too — they
    do not go through CredentialStore.load().
    """
    def test_direct_construction_https_accepted(self):
        creds = WordPressCredentials(
            url="https://overheaddoordenver.com",
            user="editor",
            app_password="xxxx xxxx xxxx xxxx xxxx xxxx",
        )
        assert creds.url == "https://overheaddoordenver.com"

    def test_direct_construction_http_raises_validation_error(self):
        with pytest.raises(ValidationError):
            WordPressCredentials(
                url="http://overheaddoordenver.com",
                user="editor",
                app_password="xxxx xxxx xxxx xxxx xxxx xxxx",
            )

    def test_model_validate_dict_http_raises(self):
        with pytest.raises(ValidationError):
            WordPressCredentials.model_validate({
                "url": "http://overheaddoordenver.com",
                "user": "editor",
                "app_password": "xxxx xxxx xxxx xxxx xxxx xxxx",
            })

    def test_model_validate_json_http_raises(self):
        import json
        raw = json.dumps({
            "url": "http://overheaddoordenver.com",
            "user": "editor",
            "app_password": "xxxx xxxx xxxx xxxx xxxx xxxx",
        })
        with pytest.raises(ValidationError):
            WordPressCredentials.model_validate_json(raw)
