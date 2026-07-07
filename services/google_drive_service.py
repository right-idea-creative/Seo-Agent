from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_IMAGE_MIMETYPES: frozenset[str] = frozenset({
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/tiff",
    "image/heic",
    "image/heif",
    "image/bmp",
})

_FOLDER_MIMETYPE = "application/vnd.google-apps.folder"

# Drive API page size: always request the maximum to minimize round-trips.
_PAGE_SIZE = 1000


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DriveFileInfo:
    """Metadata for a single image file returned from the Drive API."""
    file_id: str
    name: str
    mime_type: str
    folder_path: str      # slash-separated path from the root folder, e.g. "Garage/Residential/"
    size: int | None
    modified_at: str | None   # ISO 8601 from Drive API
    thumbnail_link: str | None
    description: str | None


@dataclass(frozen=True)
class DriveListResult:
    """Result of a full recursive Drive listing."""
    images: list[DriveFileInfo]
    folders_scanned: int
    ignored_count: int    # files that are neither images nor folders


@dataclass(frozen=True)
class FolderStats:
    """Summary of a Drive folder used for cache invalidation."""
    image_count: int
    last_modified: str | None


# ── Exceptions ────────────────────────────────────────────────────────────────

class GoogleDriveError(Exception):
    """Base exception for Drive API errors."""

class GoogleDriveAuthError(GoogleDriveError):
    """Service account credentials are invalid or lack Drive access."""

class GoogleDriveNotFoundError(GoogleDriveError):
    """The requested folder or file does not exist."""


# ── Service ───────────────────────────────────────────────────────────────────

class GoogleDriveService:
    """
    Thin client for the Google Drive API v3.

    Knows nothing about images, articles, or visual styles — it only reads
    files from Drive. All decisions about which files are relevant belong
    to callers (DriveImageIndex, VisualStyleService).

    Authentication uses a Service Account JSON key file. The service account
    must have at least Viewer access to the target folder (or Shared Drive
    membership if using a Shared Drive).

    Two approaches to image listing:
    - list_all_images()  Full recursive traversal, no limit. Returns DriveListResult
                         with folder/ignored counts. Used by DriveImageIndex.sync().
    - list_images()      Backward-compat wrapper; returns first N images from
                         list_all_images(). Used by VisualStyleService.

    Key design decision: the Drive API pageSize for list calls is always 1000
    (the API maximum), independent of how many results the caller wants. This
    prevents the previous bug where deep folder trees were silently skipped
    because pageSize shrank as results accumulated.
    """

    _SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    _FIELDS = (
        "nextPageToken, "
        "files(id, name, mimeType, size, modifiedTime, thumbnailLink, description)"
    )
    _COUNT_FIELDS = "nextPageToken, files(id, mimeType)"

    def __init__(self, service_account_path: Path) -> None:
        self._sa_path = service_account_path
        self._service: Any = None  # lazily initialized

    def _get_service(self) -> Any:
        if self._service is None:
            try:
                from google.oauth2 import service_account
                from googleapiclient.discovery import build
            except ImportError as exc:
                raise GoogleDriveError(
                    "google-api-python-client and google-auth are required for Drive integration. "
                    "Run: pip install google-api-python-client google-auth"
                ) from exc

            creds = service_account.Credentials.from_service_account_file(
                str(self._sa_path), scopes=self._SCOPES
            )
            self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return self._service

    # ── Public interface ──────────────────────────────────────────────────────

    def list_all_images(self, folder_id: str) -> DriveListResult:
        """
        Recursively list ALL images in a Drive folder tree. No limit.

        Uses pageSize=1000 for every API call to traverse the full tree
        regardless of depth or breadth, then returns a DriveListResult
        with the complete image list and diagnostic counters.

        Supports both personal Drive and Shared Drives via
        supportsAllDrives=True / includeItemsFromAllDrives=True.
        """
        images: list[DriveFileInfo] = []
        counters: dict[str, int] = {"folders": 0, "ignored": 0}
        self._collect_all(folder_id, "", images, counters)
        return DriveListResult(
            images=images,
            folders_scanned=counters["folders"],
            ignored_count=counters["ignored"],
        )

    def list_images(
        self,
        folder_id: str,
        recursive: bool = True,   # kept for backward compat — always recursive
        limit: int = 50,
    ) -> list[DriveFileInfo]:
        """
        Return the first `limit` images from the folder tree.

        Thin wrapper around list_all_images() for callers that only need
        a bounded sample (e.g. VisualStyleService visual analysis).
        """
        result = self.list_all_images(folder_id)
        return result.images[:limit]

    def get_folder_stats(self, folder_id: str) -> FolderStats:
        """
        Return image count for cache invalidation.

        Uses a minimal-fields traversal (no thumbnails, no descriptions)
        for efficiency — significantly faster than list_all_images() on
        large banks.
        """
        count = self.count_images(folder_id)
        return FolderStats(image_count=count, last_modified=None)

    def count_images(self, folder_id: str) -> int:
        """Fast recursive image count using minimal API response fields."""
        counter: dict[str, int] = {"n": 0}
        self._count_recursive(folder_id, counter)
        return counter["n"]

    def download(self, file_id: str) -> bytes:
        """Download the full content of a file by ID."""
        try:
            request = self._get_service().files().get_media(fileId=file_id)
            return request.execute()
        except Exception as exc:
            raise GoogleDriveError(f"Failed to download file {file_id}: {exc}") from exc

    def download_thumbnail(self, thumbnail_link: str, size: int = 512) -> bytes:
        """
        Download a Drive thumbnail via its thumbnailLink URL.

        Thumbnails are accessible via time-limited signed URLs included in
        file metadata. The size parameter requests a specific pixel width.
        Note: these signed URLs expire in ~1-2 hours. Prefer
        download_thumbnail_by_id() when the cached link may be stale.
        """
        url = thumbnail_link.split("=")[0] + f"=s{size}"
        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.content
        except httpx.HTTPError as exc:
            raise GoogleDriveError(f"Failed to download thumbnail: {exc}") from exc

    def download_thumbnail_by_id(self, file_id: str, size: int = 512) -> bytes:
        """
        Fetch a fresh thumbnailLink for file_id via the Drive API and download it.

        Used as a fallback when the cached thumbnailLink stored in the SQLite
        index is expired (signed URLs from Drive expire in ~1-2 hours).
        The files.get() call uses service-account auth and never expires.

        Raises GoogleDriveError if the file has no thumbnail (e.g. the file
        was recently uploaded and Drive has not generated one yet).
        """
        try:
            result = (
                self._get_service()
                .files()
                .get(
                    fileId=file_id,
                    fields="thumbnailLink",
                    supportsAllDrives=True,
                )
                .execute()
            )
        except Exception as exc:
            raise GoogleDriveError(
                f"Drive API get failed for file {file_id}: {exc}"
            ) from exc

        fresh_link = result.get("thumbnailLink")
        if not fresh_link:
            raise GoogleDriveError(
                f"No thumbnail available for file {file_id} "
                "(Drive may not have generated one yet)"
            )
        return self.download_thumbnail(fresh_link, size=size)

    # ── Internal: full traversal ──────────────────────────────────────────────

    def _collect_all(
        self,
        folder_id: str,
        path: str,           # relative path from root, e.g. "Garage Doors/Residential/"
        images: list[DriveFileInfo],
        counters: dict[str, int],
        page_token: str | None = None,
    ) -> None:
        """
        Recursively collect all images from a folder and its subfolders.

        Uses pageSize=1000 (Drive API maximum) unconditionally so that the
        number of accumulated results never reduces the page window — the
        previous implementation used min(100, limit - len(results)) which
        caused subfolders to be skipped once results were nearly full.
        """
        counters["folders"] += 1
        current_token = page_token

        while True:
            try:
                response = (
                    self._get_service()
                    .files()
                    .list(
                        q=f"'{folder_id}' in parents and trashed = false",
                        fields=self._FIELDS,
                        pageSize=_PAGE_SIZE,
                        pageToken=current_token,
                        orderBy="modifiedTime desc",
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                    )
                    .execute()
                )
            except Exception as exc:
                raise GoogleDriveError(
                    f"Drive API list failed for folder {folder_id}: {exc}"
                ) from exc

            for f in response.get("files", []):
                mime = f.get("mimeType", "")

                if mime in _IMAGE_MIMETYPES:
                    images.append(DriveFileInfo(
                        file_id=f["id"],
                        name=f["name"],
                        mime_type=mime,
                        folder_path=path,
                        size=int(f["size"]) if f.get("size") else None,
                        modified_at=f.get("modifiedTime"),
                        thumbnail_link=f.get("thumbnailLink"),
                        description=f.get("description"),
                    ))

                elif mime == _FOLDER_MIMETYPE:
                    subfolder_path = f"{path}{f['name']}/" if path else f"{f['name']}/"
                    self._collect_all(f["id"], subfolder_path, images, counters)

                else:
                    counters["ignored"] += 1

            next_token = response.get("nextPageToken")
            if not next_token:
                break
            current_token = next_token

    # ── Internal: count traversal ─────────────────────────────────────────────

    def _count_recursive(
        self,
        folder_id: str,
        counter: dict[str, int],
        page_token: str | None = None,
    ) -> None:
        """Traverse the folder tree counting images. Uses minimal API fields."""
        current_token = page_token

        while True:
            try:
                response = (
                    self._get_service()
                    .files()
                    .list(
                        q=f"'{folder_id}' in parents and trashed = false",
                        fields=self._COUNT_FIELDS,
                        pageSize=_PAGE_SIZE,
                        pageToken=current_token,
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                    )
                    .execute()
                )
            except Exception as exc:
                raise GoogleDriveError(
                    f"Drive API count failed for folder {folder_id}: {exc}"
                ) from exc

            for f in response.get("files", []):
                mime = f.get("mimeType", "")
                if mime in _IMAGE_MIMETYPES:
                    counter["n"] += 1
                elif mime == _FOLDER_MIMETYPE:
                    self._count_recursive(f["id"], counter)

            next_token = response.get("nextPageToken")
            if not next_token:
                break
            current_token = next_token
