import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class CredentialError(Exception):
    """Base exception for credential store errors."""

class CredentialNotFoundError(CredentialError):
    """No credential file exists for the given client_id / website_id pair."""

class CredentialInvalidError(CredentialError):
    """Credential file exists but is malformed or missing required fields."""


# ── Credential model ──────────────────────────────────────────────────────────

class WordPressCredentials(BaseModel):
    """
    WordPress connection details for a single site.

    Loaded from credentials/{client_id}/{website_id}.json.
    Never read from environment variables — each site has its own file.
    """
    url: str = Field(description="WordPress site URL, e.g. 'https://tu-sitio.com'.")
    user: str = Field(description="WordPress username with publish permissions.")
    app_password: str = Field(
        description="WordPress Application Password (Settings → Your Profile)."
    )
    default_category_id: int | None = Field(
        default=None,
        description=(
            "Fallback category ID used when the agent-suggested category does not "
            "exist in WordPress. Null means publish without a category."
        ),
    )


# ── Store ─────────────────────────────────────────────────────────────────────

class CredentialStore:
    """
    Loads WordPress credentials from the filesystem by tenant identity.

    File layout:
        {credentials_dir}/{client_id}/{website_id}.json

    The client_id and website_id values come from TenantContext and must
    match the directory / filename exactly (case-sensitive on Linux).

    This class has no state beyond the base directory. It reads the file
    on every load() call — no caching. Credential updates take effect
    immediately without restarting the process.
    """

    def __init__(self, credentials_dir: Path) -> None:
        self._dir = credentials_dir

    def load(self, client_id: str, website_id: str) -> WordPressCredentials:
        """
        Load and validate credentials for the given tenant.

        Args:
            client_id:  Client identifier matching TenantContext.client_id.
            website_id: Website identifier matching TenantContext.website_id.

        Returns:
            Validated WordPressCredentials ready to pass to WordPressService.

        Raises:
            CredentialNotFoundError: File does not exist.
            CredentialInvalidError:  File is not valid JSON or fails validation.
        """
        path = self._dir / client_id / f"{website_id}.json"

        if not path.exists():
            raise CredentialNotFoundError(
                f"No credentials found for client='{client_id}' / website='{website_id}'.\n"
                f"Expected file: {path}\n"
                f"See credentials/README.md for setup instructions."
            )

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CredentialInvalidError(
                f"Credential file is not valid JSON: {path}\n{exc}"
            ) from exc

        try:
            creds = WordPressCredentials.model_validate(raw)
        except ValidationError as exc:
            raise CredentialInvalidError(
                f"Credential file has missing or invalid fields: {path}\n"
                + "\n".join(f"  - {e['loc']}: {e['msg']}" for e in exc.errors())
            ) from exc

        logger.debug("Loaded credentials for %s/%s from %s", client_id, website_id, path)
        return creds

    def exists(self, client_id: str, website_id: str) -> bool:
        """Return True if a credential file already exists for the given tenant."""
        return (self._dir / client_id / f"{website_id}.json").exists()

    def save(self, client_id: str, website_id: str, credentials: WordPressCredentials) -> Path:
        """
        Persist credentials for the given tenant.

        Args:
            client_id:   Client identifier matching TenantContext.client_id.
            website_id:  Website identifier matching TenantContext.website_id.
            credentials: Validated WordPressCredentials to write.

        Returns:
            Path to the written credential file.
        """
        path = self._dir / client_id / f"{website_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(credentials.model_dump_json(indent=2), encoding="utf-8")
        logger.debug("Saved credentials for %s/%s to %s", client_id, website_id, path)
        return path
