"""
PublicationReadinessService — the single gate that decides whether an article
may be published to WordPress.

All validation rules live here. No other component evaluates production
readiness. If the gate returns ready=False, no WordPress API call is made,
and no history is updated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from models.article import Article
    from models.seo_report import SEOReport


@dataclass
class ReadinessCheck:
    name: str
    passed: bool
    detail: str = ""
    blocking: bool = True  # False → failure is a warning, not a publication blocker


@dataclass
class ReadinessResult:
    checks: list[ReadinessCheck] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return all(c.passed for c in self.checks if c.blocking)

    @property
    def failures(self) -> list[ReadinessCheck]:
        return [c for c in self.checks if not c.passed and c.blocking]

    @property
    def warnings(self) -> list[ReadinessCheck]:
        return [c for c in self.checks if not c.passed and not c.blocking]


class PublicationReadinessService:
    """
    Validates every production requirement before any WordPress API call is made.

    Operates exclusively on data already collected by the pipeline — never calls
    any external service.

    Checks run in order; all checks execute regardless of earlier failures so
    the complete picture is always available for display.
    """

    # Image placeholder patterns that must not survive into production
    _PLACEHOLDER_RES: list[re.Pattern] = [
        re.compile(r'\[IMAGE_\d+\]', re.IGNORECASE),
        re.compile(r'\[\[IMAGE[^\]]*\]\]', re.IGNORECASE),
        re.compile(r'<!--\s*IMAGE\s*-->', re.IGNORECASE),
        re.compile(r'\[INSERT_IMAGE[^\]]*\]', re.IGNORECASE),
        re.compile(r'\[PLACEHOLDER[^\]]*\]', re.IGNORECASE),
        re.compile(r'\[IMAGE HERE\]', re.IGNORECASE),
    ]

    # Standalone UI artifact patterns (conservative — full-line match only)
    _UI_ARTIFACT_RES: list[re.Pattern] = [
        re.compile(r'^\s*call\s+now[!.]?\s*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^\s*call\s+us\s+(now|today)[!.]?\s*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^\s*book\s+now[!.]?\s*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^\s*book\s+online[!.]?\s*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^\s*request\s+(a\s+)?service[!.]?\s*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^\s*contact\s+us\s+(now|today)[!.]?\s*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^\s*get\s+(a\s+)?free\s+(?:quote|estimate)[!.]?\s*$', re.IGNORECASE | re.MULTILINE),
    ]

    def validate(
        self,
        *,
        article: "Article",
        image_plan: Any | None,
        resolved_images: list,
        uploaded_images: list | None,
        links_added: int,
        no_links: bool,
        seo_qa_report: "SEOReport | None",
        min_seo_score: int,
        dual_qa_passed: bool | None,
        min_word_count: int,
    ) -> ReadinessResult:
        """
        Run all production readiness checks.

        Args:
            article:          Final article (with image markers embedded).
            image_plan:       Image resolution plan, or None when images skipped.
            resolved_images:  (ImageRequest, ImageAsset) pairs from resolver.
            uploaded_images:  (ImageRequest, ImageMetadata) pairs from upload, or None.
            links_added:      Number of internal links added by link enricher.
            no_links:         True when --no-links was passed (skips link check).
            seo_qa_report:    Rule-based SEO QA report (from SEOQAService).
            min_seo_score:    Minimum passing SEO score.
            dual_qa_passed:   Result of the Dual QA review, or None if disabled.
            min_word_count:   Minimum word count from settings.

        Returns:
            ReadinessResult — ready=True only if every check passes.
        """
        result = ReadinessResult()
        self._check_content(result, article)
        self._check_word_count(result, article, min_word_count)
        self._check_seo_qa(result, seo_qa_report, min_seo_score)
        self._check_dual_qa(result, dual_qa_passed)
        self._check_images(result, image_plan, resolved_images, uploaded_images)
        self._check_no_placeholders(result, article.content_markdown)
        self._check_internal_links(result, links_added, no_links)
        self._check_no_ui_artifacts(result, article.content_markdown)
        return result

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_content(self, result: ReadinessResult, article: "Article") -> None:
        missing = []
        if not article.title.strip():
            missing.append("title")
        if not article.seo.slug.strip():
            missing.append("slug")
        if not article.content_markdown.strip():
            missing.append("article body")
        passed = not missing
        result.checks.append(ReadinessCheck(
            "Content",
            passed,
            "OK" if passed else f"Missing: {', '.join(missing)}",
        ))

    def _check_word_count(
        self,
        result: ReadinessResult,
        article: "Article",
        minimum: int,
    ) -> None:
        # Strip markdown comment markers and formatting for an accurate count
        text = re.sub(r'<!--.*?-->', '', article.content_markdown, flags=re.DOTALL)
        text = re.sub(r'[#*_`\[\]|]', ' ', text)
        words = len(text.split())
        passed = words >= minimum
        result.checks.append(ReadinessCheck(
            "Word Count",
            passed,
            f"{words:,} words (minimum: {minimum:,})" if not passed else f"{words:,} words",
        ))

    def _check_seo_qa(
        self,
        result: ReadinessResult,
        report: "SEOReport | None",
        min_score: int,
    ) -> None:
        if report is None:
            result.checks.append(ReadinessCheck("SEO QA", False, "SEO QA report unavailable"))
            return
        passed = report.summary.critical == 0 and report.score >= min_score
        detail = f"Score {report.score}/100"
        if report.summary.critical:
            detail += f" — {report.summary.critical} critical issue(s)"
        elif not passed:
            detail += f" (minimum: {min_score})"
        result.checks.append(ReadinessCheck("SEO QA", passed, detail))

    def _check_dual_qa(self, result: ReadinessResult, passed: bool | None) -> None:
        if passed is None:
            return  # Dual QA disabled — check not applicable
        result.checks.append(ReadinessCheck(
            "Dual QA Review",
            passed,
            "All reviewers approved" if passed else "Review failed — article did not meet quality threshold",
        ))

    def _check_images(
        self,
        result: ReadinessResult,
        image_plan: Any | None,
        resolved_images: list,
        uploaded_images: list | None,
    ) -> None:
        from models.image_request import ImagePurpose

        if image_plan is None:
            # Images intentionally skipped or Drive not configured
            result.checks.append(ReadinessCheck(
                "Images", True, "Image resolution not configured (skipped)"
            ))
            return

        requested = len(image_plan.requests)
        resolved = len(resolved_images)
        uploaded = len(uploaded_images) if uploaded_images else 0

        # All requested images must be resolved (Drive + OpenAI fallback).
        # Non-blocking — resolution failures (timeouts, Drive unavailable) are warnings.
        all_resolved = resolved >= requested
        result.checks.append(ReadinessCheck(
            "Images Resolved",
            all_resolved,
            f"Requested: {requested}  Resolved: {resolved}"
            + ("" if all_resolved else f"  Missing: {requested - resolved}"),
            blocking=False,
        ))

        # Featured image must exist.
        # Non-blocking — image upload failures (timeouts, WP media API errors) are warnings.
        has_featured = any(
            req.purpose == ImagePurpose.FEATURED and bool(meta.wordpress_media_id)
            for req, meta in (uploaded_images or [])
        )
        result.checks.append(ReadinessCheck(
            "Featured Image",
            has_featured,
            "Uploaded to WordPress" if has_featured else "No featured image uploaded",
            blocking=False,
        ))

        # Every resolved image must upload successfully.
        # Non-blocking — upload failures (timeouts, HTTP errors) are warnings.
        all_uploaded = uploaded >= resolved
        result.checks.append(ReadinessCheck(
            "Image Upload",
            all_uploaded,
            f"Resolved: {resolved}  Uploaded: {uploaded}"
            + ("" if all_uploaded else f"  Failed: {resolved - uploaded}"),
            blocking=False,
        ))

    def _check_no_placeholders(self, result: ReadinessResult, markdown: str) -> None:
        found: list[str] = []
        for pat in self._PLACEHOLDER_RES:
            found.extend(pat.findall(markdown))
        passed = not found
        result.checks.append(ReadinessCheck(
            "No Placeholders",
            passed,
            "None found" if passed else f"Found: {', '.join(dict.fromkeys(found[:5]))}",
        ))

    def _check_internal_links(
        self,
        result: ReadinessResult,
        links_added: int,
        no_links: bool,
    ) -> None:
        if no_links:
            result.checks.append(ReadinessCheck(
                "Internal Links", True, "Skipped (--no-links)"
            ))
            return
        passed = links_added >= 1
        result.checks.append(ReadinessCheck(
            "Internal Links",
            passed,
            f"{links_added} link(s) inserted" if passed else "No internal links inserted by link enricher",
            blocking=False,
        ))

    def _check_no_ui_artifacts(self, result: ReadinessResult, markdown: str) -> None:
        detected: list[str] = []
        for pat in self._UI_ARTIFACT_RES:
            if pat.search(markdown):
                detected.append(pat.pattern)
        passed = not detected
        result.checks.append(ReadinessCheck(
            "No UI Artifacts",
            passed,
            "None found" if passed else f"Standalone UI elements detected: {len(detected)}",
        ))
