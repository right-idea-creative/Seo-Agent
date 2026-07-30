from __future__ import annotations

import base64
import json
import logging
import time

import services.call_tracer as call_tracer

logger = logging.getLogger(__name__)

_TEXT_REVIEW_SYSTEM = """\
You are a Senior Human Editor and Authenticity Reviewer for a premium content marketing agency.
You have reviewed thousands of articles and know exactly how AI-generated content looks, sounds,
and feels.

YOUR RESPONSIBILITY: Evaluate whether this article reads as though it was written by an
experienced local content marketer — not an AI.

You are NOT responsible for SEO. Focus entirely on:

HUMAN WRITING QUALITY (writing_score 0–100)
• Sentence rhythm: does it vary naturally, or is every sentence the same length and structure?
• Transitions: organic and context-specific, or formulaic ("Furthermore", "Additionally")?
• Voice and personality: does the writing have a distinct, consistent tone?
• Regional grounding: does it demonstrate specific knowledge of the target area — climate
  patterns, neighborhood names, housing styles, typical homeowner situations, local regulations?
  Generic city mentions with no regional texture lower this score.
• Practical specificity: does it give readers concrete, actionable information — realistic cost
  ranges, specific process steps, actual material names, real timing expectations?
  Vague generalities lower this score; useful detail raises it.
• Introduction: does it hook immediately with something relevant and direct?
• Conclusion: does it close naturally, not with "In conclusion, we have seen..."?
• Paragraph variation: different lengths, some short for emphasis?

IMPORTANT — DO NOT REQUEST IN REVISION INSTRUCTIONS:
Personal anecdotes, customer stories, fabricated testimonials, invented job examples,
or made-up statistics. These are not part of the editorial format and must never be
requested. When authenticity is low, request instead: more specific local context
(real, verifiable), stronger sentence variety, more distinctive voice, removal of
AI-pattern phrases, and more concrete practical information.

AI AUTHENTICITY CHECK (authenticity_score 0–100 — how undetectable AI is)
Red flags that LOWER this score:
• "In today's [world/landscape/digital age]..." opener
• Overuse of: Furthermore, Moreover, Additionally, It's worth noting, It's important to note,
  Needless to say, First and foremost, When it comes to, In conclusion, At the end of the day
• Formulaic 3–5 point lists where every bullet follows the same syntactic pattern
• Generic statements applicable to any company in any city anywhere
• Perfect but lifeless prose — technically correct, emotionally inert
• Repetitive sentence structure across paragraphs
• Unnatural enthusiasm or hedging ("It's absolutely critical that...")
• City name dropped without genuine local knowledge — neighborhood names, climate, housing
  characteristics, and regional context are the evidence of real local expertise
• Absence of imperfection — human writers occasionally start sentences with "And" or "But"

What RAISES this score (signals genuine expertise without fabrication):
• Specific local climate, weather, or seasonal conditions relevant to the service
• Named neighborhoods, districts, or housing types common in the target city
• Regional building codes, permit norms, or material choices specific to the area
• Practical knowledge of how homeowners in that area typically encounter this problem
• Concrete process knowledge — how a professional actually does this work, in what order,
  using what materials, at what realistic cost range

SCORING GUIDE:
95–100 = Indistinguishable from an expert human writer — would fool any editor
90–94  = Very strong; only a trained AI-content reviewer might notice
80–89  = Mostly natural but with identifiable AI patterns in places
70–79  = Clearly assisted — AI patterns visible throughout
< 70   = Obviously AI-generated

DECISION RULE:
approved = true ONLY if writing_score >= 90 AND authenticity_score >= 90.
Be strict. The bar is high. Mediocre is not good enough.

Return valid JSON only — no prose before or after the object.\
"""

_VISION_REVIEW_SYSTEM = """\
You are a Quality Control Inspector for a local service business media team.
You evaluate edited company photographs for identity preservation before publication.

PRIMARY QUESTION — answer this FIRST before scoring anything else:
"Does this still look like the SAME original company photograph?"

When the original photograph is provided for comparison, PRIORITIZE:
• Are the same technician, uniform, face, hands, and equipment present in both?
• Is the same house, garage door, driveway, and neighborhood visible in both?
• Is the same truck (if present) visible in both?
• Does the overall composition remain the same?
• Do approximately 90–100% of the original pixels appear unchanged?

EDIT QUALITY (only if identity is preserved)
• Was the edit minimal and surgical — not a full scene replacement?
• Are there visible artifacts, seams, or warped geometry from the edit?
• Is the edit cleanly integrated with the original photograph?

SCORING GUIDE:
95–100 = Same photograph — edit is invisible or nearly so; no artifacts
90–94  = Same photograph with a clearly visible but clean and appropriate edit
80–89  = Same photograph but edit has minor artifacts or looks slightly unnatural
70–79  = Questionable — core elements mostly there but some deviations noticed
< 70   = Identity compromised — REJECT

DECISION RULE:
approved = true ONLY if vision_score >= 90 AND the image is still recognizably the same photograph.
If in doubt, reject — the original Drive photo is always the safe fallback.

Return valid JSON only — no prose before or after the object.\
"""


class OpenAIReviewService:
    """
    Second-opinion reviewer using OpenAI GPT models.

    Text review (review_article): evaluates writing naturalness and AI-detection resistance.
    Vision review (review_image): evaluates AI-generated images for visual artifacts and realism.

    This is the second independent reviewer in the dual-QA pipeline.
    Claude acts as SEO/editorial reviewer; OpenAI acts as human authenticity reviewer.
    Each has entirely different responsibilities — neither checks what the other checks.

    Cost tracking: accumulates USD cost estimates from API response usage. Accessible
    via text_cost_usd and vision_cost_usd for inclusion in the QA cost report.
    """

    # OpenAI pricing (per million tokens) — keyed by model name.
    _PRICING: dict[str, tuple[float, float]] = {
        "gpt-4o":      (2.50, 10.00),
        "gpt-4o-mini": (0.15,  0.60),
    }

    def __init__(
        self,
        api_key: str,
        text_model: str = "gpt-4o",
        vision_model: str = "gpt-4o",
    ) -> None:
        try:
            import openai as _openai
            self._client = _openai.OpenAI(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "openai package is required for dual QA. Run: pip install openai"
            ) from exc
        self._text_model = text_model
        self._vision_model = vision_model
        self.text_cost_usd: float = 0.0    # cumulative text review cost
        self.vision_cost_usd: float = 0.0  # cumulative vision review cost

    def review_article(self, article_text: str, seo_context: str) -> dict:
        """
        Review an article for human writing quality and authenticity.

        Returns a dict with:
          writing_score (int 0–100)
          authenticity_score (int 0–100)
          approved (bool)
          writing_feedback (str)
          authenticity_feedback (str)
          issues (list[str])
          revision_instructions (str)
        """
        user_content = (
            f"SEO CONTEXT (for background only — do not evaluate SEO):\n{seo_context}\n\n"
            f"ARTICLE TO REVIEW:\n{article_text}\n\n"
            "Return a JSON object with EXACTLY these fields:\n"
            '{\n'
            '  "writing_score": <integer 0-100>,\n'
            '  "writing_reasoning": "<1-3 sentences explaining this score>",\n'
            '  "writing_strengths": ["<what works well 1>", "<what works well 2>"],\n'
            '  "writing_weaknesses": ["<specific problem 1>", "<specific problem 2>"],\n'
            '  "writing_improvements": ["<concrete fix 1>", "<concrete fix 2>"],\n'
            '  "writing_priority": "<High|Medium|Low — how urgently must this improve?>",\n'
            '  "authenticity_score": <integer 0-100>,\n'
            '  "authenticity_reasoning": "<1-3 sentences explaining this score>",\n'
            '  "authenticity_strengths": ["<what passes as human 1>"],\n'
            '  "authenticity_weaknesses": ["<AI tell-tale 1>", "<AI tell-tale 2>"],\n'
            '  "authenticity_improvements": ["<how to mask pattern 1>", "<how to mask pattern 2>"],\n'
            '  "authenticity_priority": "<High|Medium|Low>",\n'
            '  "approved": <true|false>,\n'
            '  "writing_feedback": "<detailed feedback on writing quality>",\n'
            '  "authenticity_feedback": "<detailed feedback on AI authenticity>",\n'
            '  "issues": ["<specific issue 1>", "<specific issue 2>"],\n'
            '  "revision_instructions": "<concrete, specific instructions to fix issues>"\n'
            '}'
        )

        t0 = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._text_model,
                messages=[
                    {"role": "system", "content": _TEXT_REVIEW_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=2048,
            )
            duration = time.perf_counter() - t0
            if response.usage:
                cost = self._compute_cost(
                    self._text_model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )
                self.text_cost_usd += cost
                call_tracer.record(
                    stage="openai:text-review",
                    model=self._text_model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    duration_s=duration,
                )
            raw = json.loads(response.choices[0].message.content or "{}")
            return self._normalize_article_review(raw)
        except Exception as exc:
            logger.error("OpenAI article review failed: %s", exc)
            raise

    def review_image(
        self,
        image_bytes: bytes,
        context_description: str,
        mime_type: str = "image/png",
        original_image: bytes | None = None,
    ) -> dict:
        """
        Review an edited company photograph for identity preservation.

        When original_image is provided, it is shown first so the model can compare
        the original against the edited version and evaluate identity preservation.
        Primary question: "Does this still look like the SAME original photograph?"

        Returns a dict with:
          vision_score (int 0–100)
          approved (bool)
          feedback (str)
          ai_artifacts_found (list[str])
          revision_instructions (str)
        """
        user_content: list[dict] = []

        if original_image is not None:
            orig_b64 = base64.b64encode(original_image).decode("utf-8")
            user_content.append({
                "type": "text",
                "text": (
                    "Here is the ORIGINAL company photograph before any edit was applied. "
                    "This is the source of truth — every visual element must be preserved:"
                ),
            })
            user_content.append({
                "type": "text",
                "text": "Original photograph:",
            })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}", "detail": "low"},
            })
            user_content.append({
                "type": "text",
                "text": (
                    "Now here is the edited version. "
                    "Does this still look like the SAME original company photograph?"
                ),
            })

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"

        user_content.extend([
            {
                "type": "text",
                "text": (
                    f"CONTEXT: {context_description}\n\n"
                    "Evaluate this image and return a JSON object with EXACTLY these fields:\n"
                    '{\n'
                    '  "vision_score": <integer 0-100>,\n'
                    '  "approved": <true|false>,\n'
                    '  "feedback": "<overall assessment of authenticity and identity match>",\n'
                    '  "ai_artifacts_found": ["<artifact 1>", "<artifact 2>"],\n'
                    '  "revision_instructions": "<how to fix if rejected>"\n'
                    '}'
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": data_url, "detail": "high"},
            },
        ])

        t0 = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._vision_model,
                messages=[
                    {"role": "system", "content": _VISION_REVIEW_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=1024,
            )
            duration = time.perf_counter() - t0
            if response.usage:
                cost = self._compute_cost(
                    self._vision_model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )
                self.vision_cost_usd += cost
                call_tracer.record(
                    stage="openai:vision-review",
                    model=self._vision_model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    duration_s=duration,
                )
            raw = json.loads(response.choices[0].message.content or "{}")
            return self._normalize_image_review(raw)
        except Exception as exc:
            logger.error("OpenAI vision review failed: %s", exc)
            raise

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        in_p, out_p = self._PRICING.get(model, (2.50, 10.00))
        return round(
            input_tokens * in_p / 1_000_000
            + output_tokens * out_p / 1_000_000,
            6,
        )

    @staticmethod
    def _normalize_article_review(raw: dict) -> dict:
        return {
            "writing_score": int(raw.get("writing_score") or 0),
            "writing_reasoning": str(raw.get("writing_reasoning", "")),
            "writing_strengths": list(raw.get("writing_strengths", [])),
            "writing_weaknesses": list(raw.get("writing_weaknesses", [])),
            "writing_improvements": list(raw.get("writing_improvements", [])),
            "writing_priority": str(raw.get("writing_priority", "")),
            "authenticity_score": int(raw.get("authenticity_score") or 0),
            "authenticity_reasoning": str(raw.get("authenticity_reasoning", "")),
            "authenticity_strengths": list(raw.get("authenticity_strengths", [])),
            "authenticity_weaknesses": list(raw.get("authenticity_weaknesses", [])),
            "authenticity_improvements": list(raw.get("authenticity_improvements", [])),
            "authenticity_priority": str(raw.get("authenticity_priority", "")),
            "approved": bool(raw.get("approved", False)),
            "writing_feedback": str(raw.get("writing_feedback", "")),
            "authenticity_feedback": str(raw.get("authenticity_feedback", "")),
            "issues": list(raw.get("issues", [])),
            "revision_instructions": str(raw.get("revision_instructions", "")),
        }

    @staticmethod
    def _normalize_image_review(raw: dict) -> dict:
        return {
            "vision_score": int(raw.get("vision_score") or 0),
            "approved": bool(raw.get("approved", False)),
            "feedback": str(raw.get("feedback", "")),
            "ai_artifacts_found": list(raw.get("ai_artifacts_found", [])),
            "revision_instructions": str(raw.get("revision_instructions", "")),
        }
