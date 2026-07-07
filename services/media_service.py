from __future__ import annotations

import logging

from models.image_asset import ImageAsset
from models.media import ImageMetadata
from services.wordpress_service import WordPressService

logger = logging.getLogger(__name__)


class MediaService:
    """
    Uploads images to the WordPress Media Library.

    Single responsibility: receive an ImageAsset (pre-upload, in-memory) and
    return an ImageMetadata (post-upload, with WordPress IDs and URLs).

    Knows nothing about where the image came from (Drive, AI, URL) — that is
    ImageResolverAgent's concern. Knows nothing about posts or publishing —
    that is PublisherAgent's concern.

    Uses WordPressService for all HTTP communication, inheriting its auth,
    error handling, and connection management without duplicating any of it.
    """

    def __init__(self, wp: WordPressService) -> None:
        self._wp = wp

    def upload(self, asset: ImageAsset) -> ImageMetadata:
        """
        Upload an ImageAsset to the WordPress Media Library.

        Args:
            asset: In-memory image data with metadata.

        Returns:
            ImageMetadata populated with WordPress media_id, source_url,
            dimensions, and the alt_text / caption from the asset.

        Raises:
            WordPressAuthError:  Insufficient permissions for media upload.
            WordPressAPIError:   Any other API or network error.
        """
        logger.info("Uploading '%s' (%s) to WP Media Library...", asset.filename, asset.mime_type)

        response = self._wp.upload_media(
            filename=asset.filename,
            data=asset.data,
            mime_type=asset.mime_type,
            alt_text=asset.alt_text,
            caption=asset.caption or "",
        )

        media_details = response.get("media_details", {})
        width  = media_details.get("width")
        height = media_details.get("height")

        return ImageMetadata(
            original_filename=asset.filename,
            alt_text=asset.alt_text,
            caption=asset.caption,
            url=response.get("source_url"),
            wordpress_media_id=response.get("id"),
            uploaded_filename=response.get("slug"),
            width=int(width) if width else None,
            height=int(height) if height else None,
            mime_type=asset.mime_type,
            source_keyword=None,
        )
