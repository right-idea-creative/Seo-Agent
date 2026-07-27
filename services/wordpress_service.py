import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from models.enums import SEOPlugin
from services.credential_store import WordPressCredentials

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class WordPressError(Exception):
    """Base exception for all WordPress service errors."""

class WordPressAuthError(WordPressError):
    """Raised on 401 / 403 — bad credentials or insufficient permissions."""

class WordPressNotFoundError(WordPressError):
    """Raised on 404 — the requested resource does not exist."""

class WordPressAPIError(WordPressError):
    """Raised on any other HTTP or connectivity error."""


# ── Validation result ─────────────────────────────────────────────────────────

@dataclass
class SiteValidationResult:
    """Encapsulates the outcome of WordPressService.validate_site()."""
    rest_api_reachable: bool = False
    auth_ok: bool = False
    auth_user: str = ""
    seo_plugin: SEOPlugin = SEOPlugin.NONE
    agent_plugin_installed: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """True when all required components are confirmed active."""
        return self.rest_api_reachable and self.auth_ok and self.agent_plugin_installed

    @property
    def status(self) -> str:
        """Single source of truth for site readiness: READY | READY_WITH_WARNINGS | FAILED."""
        if not self.rest_api_reachable or not self.auth_ok:
            return "FAILED"
        if not self.agent_plugin_installed or self.seo_plugin == SEOPlugin.NONE:
            return "READY_WITH_WARNINGS"
        return "READY"


# ── Service ───────────────────────────────────────────────────────────────────

class WordPressService:
    """
    Thin HTTP client for the WordPress REST API.

    Knows nothing about Article, SEO, or publishing strategy — those
    decisions belong to PublisherAgent. This class only knows how to
    translate Python calls into WP REST API requests.

    Implements the context manager protocol so httpx.Client is closed
    cleanly after a publish run:

        with WordPressService(url, user, password) as wp:
            publisher = PublisherAgent(wp)
            ...
    """

    _API_PATH = "/wp-json/wp/v2"

    def __init__(self, credentials: WordPressCredentials) -> None:
        self._credentials = credentials
        self._base = credentials.url.rstrip("/") + self._API_PATH
        self._client = httpx.Client(
            auth=(credentials.user, credentials.app_password),
            timeout=30,
            follow_redirects=True,  # handle HTTP → HTTPS redirects transparently
        )

    @property
    def site_url(self) -> str:
        """Base URL of the WordPress site (no trailing slash)."""
        return self._credentials.url.rstrip("/")

    @property
    def default_category_id(self) -> int | None:
        """Fallback category ID from the site credential file."""
        return self._credentials.default_category_id

    def __enter__(self) -> "WordPressService":
        return self

    def __exit__(self, *args: object) -> None:
        self._client.close()

    # ── Connectivity ──────────────────────────────────────────────────────────

    def check_connection(self) -> None:
        """
        Verify the WP REST API is reachable.

        Raises WordPressAPIError if the endpoint does not respond.
        Does NOT verify credentials — use check_auth() for that.
        """
        self._get("/types")

    def check_auth(self) -> str:
        """
        Verify credentials are valid.

        Returns:
            Display name of the authenticated WordPress user.

        Raises:
            WordPressAuthError: Credentials are invalid or insufficient.
        """
        data = self._get("/users/me")
        return data.get("name", data.get("slug", "unknown"))

    def detect_seo_plugin(self) -> SEOPlugin:
        """
        Auto-detect the active SEO plugin by inspecting WP REST API namespaces.

        Checks the public /wp-json/ index for known plugin namespace prefixes.
        Returns SEOPlugin.NONE if detection fails or no known plugin is found.
        """
        try:
            root_url = self._credentials.url.rstrip("/") + "/wp-json/"
            resp = self._client.get(root_url, timeout=10)
            if resp.status_code == 200:
                namespaces: list[str] = resp.json().get("namespaces", [])
                if any(ns.startswith("yoast") for ns in namespaces):
                    logger.info("SEO plugin detected: Yoast SEO")
                    return SEOPlugin.YOAST
                if any("rankmath" in ns for ns in namespaces):
                    logger.info("SEO plugin detected: Rank Math")
                    return SEOPlugin.RANKMATH
        except Exception as exc:
            logger.warning("SEO plugin auto-detection failed: %s", exc)
        logger.info("No known SEO plugin detected — meta fields will not be sent.")
        return SEOPlugin.NONE

    def validate_site(self) -> SiteValidationResult:
        """
        Run a complete site health check and return a structured result.

        Checks REST API reachability, credential validity, SEO plugin presence,
        and seo-agent.php plugin presence. Short-circuits on the first fatal
        failure (unreachable or bad auth) so errors are unambiguous.
        """
        result = SiteValidationResult()

        try:
            self.check_connection()
            result.rest_api_reachable = True
        except Exception as exc:
            result.errors.append(f"REST API unreachable: {exc}")
            return result

        try:
            result.auth_user = self.check_auth()
            result.auth_ok = True
        except WordPressAuthError as exc:
            result.errors.append(f"Authentication failed: {exc}")
        except Exception as exc:
            result.errors.append(f"Auth check error: {exc}")

        namespaces = self._get_wp_namespaces()
        if any(ns.startswith("yoast") for ns in namespaces):
            result.seo_plugin = SEOPlugin.YOAST
        elif any("rankmath" in ns for ns in namespaces):
            result.seo_plugin = SEOPlugin.RANKMATH
        result.agent_plugin_installed = self._check_agent_plugin()

        if result.seo_plugin == SEOPlugin.NONE:
            result.errors.append(
                "No supported SEO plugin detected. Yoast or Rank Math is required."
            )
        if not result.agent_plugin_installed:
            result.errors.append(
                "seo-agent.php plugin not detected. "
                "Required for idempotent publishing and Yoast meta writes."
            )

        return result

    def _get_wp_namespaces(self) -> list[str]:
        """Fetch /wp-json/ and return its namespace list."""
        try:
            root_url = self._credentials.url.rstrip("/") + "/wp-json/"
            resp = self._client.get(root_url, timeout=10)
            if resp.status_code == 200:
                return resp.json().get("namespaces", [])
        except Exception as exc:
            logger.warning("Could not fetch WP-JSON root: %s", exc)
        return []

    def _check_agent_plugin(self) -> bool:
        """
        Detect seo-agent.php via the REST schema, not the namespace registry.

        The plugin registers meta fields with register_post_meta(show_in_rest=True).
        It never calls register_rest_route(), so it never appears in /wp-json/
        namespaces. The correct signal is _seo_agent_id appearing in the
        OPTIONS /wp-json/wp/v2/posts schema — a read-only, side-effect-free probe
        that works even on empty sites.
        """
        try:
            resp = self._client.options(self._base + "/posts", timeout=10)
            if resp.status_code == 200:
                meta_props = (
                    resp.json()
                    .get("schema", {})
                    .get("properties", {})
                    .get("meta", {})
                    .get("properties", {})
                )
                return "_seo_agent_id" in meta_props
        except Exception as exc:
            logger.warning("Agent plugin schema probe failed: %s", exc)
        return False

    # ── Taxonomy ──────────────────────────────────────────────────────────────

    def get_category_by_name(self, name: str) -> int | None:
        """
        Return the ID of an existing category with an exact name match.

        The search is case-insensitive. Returns None if no match is found.
        Does not create categories — that is a deliberate editorial decision.
        """
        results = self._get("/categories", params={"search": name, "per_page": 20})
        for cat in results:
            if cat["name"].strip().lower() == name.strip().lower():
                logger.debug("Category '%s' resolved to ID %d", name, cat["id"])
                return cat["id"]
        logger.warning("Category '%s' not found in WordPress", name)
        return None

    def find_tag_by_name(self, name: str) -> int | None:
        """Return the ID of an existing tag with an exact name match, or None."""
        results = self._get("/tags", params={"search": name, "per_page": 20})
        for tag in results:
            if tag["name"].strip().lower() == name.strip().lower():
                return tag["id"]
        return None

    def get_or_create_tag(self, name: str) -> int:
        """
        Return the ID of an existing tag or create it if it does not exist.

        Tags are created automatically because they are article-specific
        and do not represent editorial structure.
        """
        existing = self.find_tag_by_name(name)
        if existing is not None:
            logger.debug("Tag '%s' found (ID %d)", name, existing)
            return existing

        created = self._post("/tags", {"name": name})
        logger.debug("Tag '%s' created (ID %d)", name, created["id"])
        return created["id"]

    # ── Posts ─────────────────────────────────────────────────────────────────

    def get_post(self, post_id: int) -> dict[str, Any] | None:
        """Fetch a post by ID. Returns None if the post does not exist."""
        try:
            return self._get(f"/posts/{post_id}")
        except WordPressNotFoundError:
            return None

    def find_post_by_meta(self, key: str, value: str) -> dict[str, Any] | None:
        """
        Find a post by custom meta key/value.

        Returns None if the meta is not registered (plugin inactive),
        if no post matches, or if the search fails for any reason.
        Requires the meta field to be registered with show_in_rest=True
        (see wordpress/seo-agent/seo-agent.php).
        """
        try:
            results = self._get("/posts", params={
                "meta_key": key,
                "meta_value": value,
                "per_page": 1,
                "status": "any",
            })
            return results[0] if isinstance(results, list) and results else None
        except (WordPressAPIError, WordPressNotFoundError):
            return None

    def find_post_by_slug(self, slug: str) -> dict[str, Any] | None:
        """Find a post by slug. Returns None if not found."""
        results = self._get("/posts", params={"slug": slug, "per_page": 1, "status": "any"})
        return results[0] if isinstance(results, list) and results else None

    def list_posts(self, per_page: int = 50) -> list[dict[str, Any]]:
        """
        Fetch published posts for internal link building.

        Returns a lightweight list with only title, link, and slug fields.
        Returns an empty list on any error so link enrichment degrades gracefully.
        """
        try:
            results = self._get("/posts", params={
                "per_page": per_page,
                "status": "publish",
                "_fields": "id,title,link,slug",
            })
            return results if isinstance(results, list) else []
        except Exception as exc:
            logger.warning("Could not fetch posts for link enrichment: %s", exc)
            return []

    def verify_meta_accepted(
        self, post_id: int, meta_keys: list[str]
    ) -> dict[str, str]:
        """
        Fetch a post and report which meta keys are visible in the REST response.

        WordPress only returns meta fields that are registered with show_in_rest=True.
        Fields absent from the response were either not registered or not sent.

        Returns: {key: "accepted" | "empty" | "not_registered" | "error"}
          - "accepted":       key is present with a non-empty value → saved correctly
          - "empty":          key is registered but stored as empty string / null
          - "not_registered": key is absent from the response → show_in_rest not set
          - "error":          GET request failed — status unknown
        """
        try:
            post = self._get(f"/posts/{post_id}", params={"_fields": "id,meta"})
            meta: dict[str, Any] = post.get("meta", {})
        except Exception as exc:
            logger.warning("Could not verify meta for post %d: %s", post_id, exc)
            return {key: "error" for key in meta_keys}

        result: dict[str, str] = {}
        for key in meta_keys:
            if key not in meta:
                result[key] = "not_registered"
            elif meta[key] in (None, "", [], {}):
                result[key] = "empty"
            else:
                result[key] = "accepted"
        return result

    def create_post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Create a WordPress post.

        Returns:
            Full post object from the WP API, including 'id' and 'link'.
        """
        return self._post("/posts", payload)

    def update_post(self, post_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Update an existing WordPress post (PATCH).

        Returns:
            Updated post object from the WP API, including 'id' and 'link'.

        Raises:
            WordPressNotFoundError: Post does not exist (was deleted in WP).
        """
        return self._patch(f"/posts/{post_id}", payload)

    # ── Media library ─────────────────────────────────────────────────────────

    def upload_media(
        self,
        filename: str,
        data: bytes,
        mime_type: str,
        alt_text: str = "",
        caption: str = "",
    ) -> dict[str, Any]:
        """
        Upload a file to the WordPress Media Library.

        Uses multipart/form-data as required by the WP REST API media endpoint.
        The Content-Disposition header provides the filename; WordPress derives
        the attachment title and slug from it automatically.

        Returns:
            Media attachment object including 'id', 'source_url', and 'media_details'.

        Raises:
            WordPressAuthError:  Credentials lack upload permissions.
            WordPressAPIError:   Upload failed for any other reason.
        """
        try:
            resp = self._client.post(
                self._base + "/media",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "Content-Type": mime_type,
                },
                content=data,
            )
            self._raise_for_status(resp)
            media = resp.json()
        except httpx.HTTPError as exc:
            raise WordPressAPIError(f"Media upload failed: {exc}") from exc

        # Set alt text and caption via a PATCH after creation, since the initial
        # POST only accepts binary content and does not take JSON fields.
        media_id = media["id"]
        update: dict[str, Any] = {}
        if alt_text:
            update["alt_text"] = alt_text
        if caption:
            update["caption"] = caption
        if update:
            try:
                media = self._patch(f"/media/{media_id}", update)
            except WordPressAPIError:
                logger.warning("Could not set alt_text/caption on media ID %d", media_id)

        logger.info("Media uploaded: ID=%d URL=%s", media["id"], media.get("source_url", ""))
        return media

    # ── HTTP primitives ───────────────────────────────────────────────────────

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = self._base + path
        try:
            resp = self._client.get(url, params=params)
            logger.debug("GET %s → HTTP %d", url, resp.status_code)
            self._raise_for_status(resp)
            return self._parse_json(resp, "GET", url)
        except httpx.HTTPError as exc:
            raise WordPressAPIError(f"GET {url} failed: {exc}") from exc

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        url = self._base + path
        try:
            resp = self._client.post(url, json=data)
            logger.debug("POST %s → HTTP %d", url, resp.status_code)
            logger.debug("POST response body: %s", resp.text[:500])
            self._raise_for_status(resp)
            result = self._parse_json(resp, "POST", url)
            return result
        except httpx.HTTPError as exc:
            raise WordPressAPIError(f"POST {url} failed: {exc}") from exc

    def _patch(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        url = self._base + path
        try:
            resp = self._client.patch(url, json=data)
            logger.debug("PATCH %s → HTTP %d", url, resp.status_code)
            self._raise_for_status(resp)
            result = self._parse_json(resp, "PATCH", url)
            return result
        except httpx.HTTPError as exc:
            raise WordPressAPIError(f"PATCH {url} failed: {exc}") from exc

    @staticmethod
    def _parse_json(resp: "httpx.Response", method: str, url: str) -> Any:
        """
        Parse response body as JSON, raising WordPressAPIError on failure.

        This prevents json.JSONDecodeError from propagating as an unhandled
        exception when WordPress (or a proxy) returns HTML or an empty body
        for a request that _raise_for_status passed silently.
        """
        try:
            return resp.json()
        except Exception as exc:
            preview = resp.text[:200].replace("\n", " ")
            raise WordPressAPIError(
                f"{method} {url} returned HTTP {resp.status_code} "
                f"but body is not valid JSON: {preview!r}"
            ) from exc

    def _raise_for_status(self, resp: "httpx.Response") -> None:
        """
        Raise the appropriate exception for any non-2xx response.

        WordPress REST API always returns 2xx on success (typically 200 or 201).
        Any other status — including 3xx redirects — is treated as an error.
        3xx without follow_redirects=True means the client URL is wrong (HTTP
        vs HTTPS mismatch, missing subdirectory, etc.).
        """
        code = resp.status_code

        # --- explicit auth errors ---
        if code in (401, 403):
            raise WordPressAuthError(
                f"Authentication failed (HTTP {code}). "
                "Check WP_USER and WP_APP_PASSWORD in .env."
            )

        # --- not found ---
        if code == 404:
            raise WordPressNotFoundError(f"Resource not found (HTTP 404): {resp.url}")

        # --- any non-2xx (including 3xx redirects and 5xx server errors) ---
        if not (200 <= code < 300):
            try:
                detail = resp.json().get("message", resp.text[:300])
            except Exception:
                detail = resp.text[:300] or f"(empty body, status {code})"
            raise WordPressAPIError(f"HTTP {code}: {detail}")

        # --- 2xx body may still contain a WP error envelope ---
        # e.g. some WP configs return 200 with {"code":"rest_forbidden",...}
        # We detect this early so callers always receive a real post object.
        try:
            body = resp.json()
        except Exception:
            return  # non-JSON 2xx; caller's _parse_json will raise properly

        if isinstance(body, dict) and "code" in body and "id" not in body:
            msg = body.get("message", str(body)[:200])
            raise WordPressAPIError(
                f"WordPress returned an error envelope (HTTP {code}): {msg}"
            )
