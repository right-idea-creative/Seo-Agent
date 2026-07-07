"""
DriveImageIndex — local SQLite index for the Drive image bank.

Separates "what images exist" (Drive API traversal) from "which image to use"
(ImageResolverAgent decisions). After the initial sync, image listings are
served entirely from the local database — no Drive API calls per article.

Sync strategy
─────────────
Full sync   Traverse all Drive folders, upsert every image, remove deleted ones.
            Triggered on first run, when folder_id changes, or when the index
            is older than max_age_hours (default 168h = 7 days).

Skipped     Index is fresh — list_all() returns local data immediately,
            with zero Drive API calls.

Schema is forward-compatible: ai_description / ai_tags / ai_category columns
are reserved for future Claude Vision enrichment and are always NULL in MVP.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from services.google_drive_service import DriveFileInfo

if TYPE_CHECKING:
    from services.google_drive_service import GoogleDriveService

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    file_id        TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    folder_path    TEXT NOT NULL DEFAULT '',
    mime_type      TEXT NOT NULL,
    size           INTEGER,
    modified_at    TEXT,
    thumbnail_link TEXT,
    description    TEXT,
    synced_at      TEXT NOT NULL,
    ai_description TEXT,
    ai_tags        TEXT,
    ai_category    TEXT
);

CREATE TABLE IF NOT EXISTS sync_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SyncStats:
    sync_type: str          # "full" | "skipped"
    folders_scanned: int
    images_found: int
    images_added: int
    images_updated: int
    images_removed: int
    ignored_files: int
    duration_seconds: float


# ── Index ─────────────────────────────────────────────────────────────────────

class DriveImageIndex:
    """
    SQLite-backed local index of all images in the Drive bank.

    Usage
    ─────
    index = DriveImageIndex(Path("index/drive_images.db"))

    if index.needs_sync(folder_id):
        stats = index.sync(drive_service, folder_id)

    candidates = index.list_all()   # list[DriveFileInfo], zero API calls
    """

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Public interface ──────────────────────────────────────────────────────

    def needs_sync(self, folder_id: str, max_age_hours: int = 168) -> bool:
        """
        True if the index needs a full sync before being used.

        Checks (in order):
        1. Database is empty or has no images.
        2. folder_id changed since last sync.
        3. Last full sync is older than max_age_hours.
        """
        meta = self._load_meta()
        if not meta:
            return True
        if meta.get("folder_id") != folder_id:
            logger.debug("Drive index: folder_id changed — full sync needed.")
            return True
        if not self._has_images():
            return True
        last_sync = meta.get("last_full_sync")
        if not last_sync:
            return True
        try:
            synced_at = datetime.fromisoformat(last_sync)
            age_hours = (datetime.now(timezone.utc) - synced_at).total_seconds() / 3600
            if age_hours > max_age_hours:
                logger.debug("Drive index: stale (%.0fh old, limit %dh).", age_hours, max_age_hours)
                return True
        except ValueError:
            return True
        return False

    def sync(self, drive: "GoogleDriveService", folder_id: str) -> SyncStats:
        """
        Full sync: traverse Drive, upsert all images, remove deleted ones.

        All changes are committed in a single transaction for atomicity.
        If the Drive API call fails, the existing index is untouched.
        """
        t0 = time.perf_counter()
        logger.info("Drive index: starting full sync for folder %s", folder_id)

        result = drive.list_all_images(folder_id)
        drive_images = result.images
        drive_ids = {img.file_id for img in drive_images}
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        try:
            existing_ids = {
                row["file_id"]
                for row in conn.execute("SELECT file_id FROM images").fetchall()
            }

            added = updated = 0
            for img in drive_images:
                if img.file_id in existing_ids:
                    conn.execute(
                        """UPDATE images
                           SET name=?, folder_path=?, mime_type=?, size=?,
                               modified_at=?, thumbnail_link=?, description=?, synced_at=?
                           WHERE file_id=?""",
                        (
                            img.name, img.folder_path, img.mime_type, img.size,
                            img.modified_at, img.thumbnail_link, img.description, now,
                            img.file_id,
                        ),
                    )
                    updated += 1
                else:
                    conn.execute(
                        """INSERT INTO images
                               (file_id, name, folder_path, mime_type, size,
                                modified_at, thumbnail_link, description, synced_at)
                           VALUES (?,?,?,?,?,?,?,?,?)""",
                        (
                            img.file_id, img.name, img.folder_path, img.mime_type,
                            img.size, img.modified_at, img.thumbnail_link,
                            img.description, now,
                        ),
                    )
                    added += 1

            stale_ids = list(existing_ids - drive_ids)
            if stale_ids:
                placeholders = ",".join("?" * len(stale_ids))
                conn.execute(
                    f"DELETE FROM images WHERE file_id IN ({placeholders})", stale_ids
                )
            removed = len(stale_ids)

            for key, value in {
                "folder_id": folder_id,
                "last_full_sync": now,
                "image_count": str(len(drive_images)),
                "folders_scanned": str(result.folders_scanned),
            }.items():
                conn.execute(
                    "INSERT OR REPLACE INTO sync_meta (key, value) VALUES (?, ?)",
                    (key, value),
                )

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        elapsed = time.perf_counter() - t0
        logger.info(
            "Drive index sync complete: %d images, %d folders, +%d/~%d/-%d in %.1fs",
            len(drive_images), result.folders_scanned, added, updated, removed, elapsed,
        )
        return SyncStats(
            sync_type="full",
            folders_scanned=result.folders_scanned,
            images_found=len(drive_images),
            images_added=added,
            images_updated=updated,
            images_removed=removed,
            ignored_files=result.ignored_count,
            duration_seconds=elapsed,
        )

    def list_all(self) -> list[DriveFileInfo]:
        """
        Return all indexed images ordered by most recently modified first.

        Zero Drive API calls — reads entirely from the local SQLite database.
        """
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """SELECT file_id, name, folder_path, mime_type, size,
                          modified_at, thumbnail_link, description
                   FROM images
                   ORDER BY modified_at DESC"""
            ).fetchall()
        finally:
            conn.close()

        return [
            DriveFileInfo(
                file_id=row["file_id"],
                name=row["name"],
                folder_path=row["folder_path"],
                mime_type=row["mime_type"],
                size=row["size"],
                modified_at=row["modified_at"],
                thumbnail_link=row["thumbnail_link"],
                description=row["description"],
            )
            for row in rows
        ]

    def stats(self) -> dict:
        """Return sync metadata from the local database (no API calls)."""
        meta = self._load_meta() or {}
        conn = sqlite3.connect(str(self._path))
        try:
            count = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        finally:
            conn.close()
        return {
            "folder_id": meta.get("folder_id"),
            "last_full_sync": meta.get("last_full_sync"),
            "image_count": count,
            "folders_scanned": int(meta.get("folders_scanned", 0)),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        conn = sqlite3.connect(str(self._path))
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    def _load_meta(self) -> dict | None:
        try:
            conn = sqlite3.connect(str(self._path))
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute("SELECT key, value FROM sync_meta").fetchall()
            finally:
                conn.close()
            return {row["key"]: row["value"] for row in rows} if rows else None
        except sqlite3.Error:
            return None

    def _has_images(self) -> bool:
        try:
            conn = sqlite3.connect(str(self._path))
            try:
                count = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            finally:
                conn.close()
            return count > 0
        except sqlite3.Error:
            return False
