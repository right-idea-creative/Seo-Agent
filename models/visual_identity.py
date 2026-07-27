from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VisualIdentityProfile:
    """
    Deep visual identity profile extracted from article-relevant Drive photographs.

    Article-specific: built from the TOP 10–20 Drive photos most semantically
    relevant to the article being produced. Every AI generation must originate
    from this profile — the AI never invents a new visual identity.

    Contrast with VisualStyleProfile (global, generic, 30-day cache):
    that model captures broad photography style.
    This model captures the SPECIFIC company identity: who their technician is,
    what their truck looks like, what neighborhoods they work in, and exactly
    how their phone camera behaves on a real service call.
    """

    # ── People ────────────────────────────────────────────────────────────────
    technician_description: str = ""
    uniform_style: str = ""

    # ── Equipment & vehicles ──────────────────────────────────────────────────
    truck_description: str = ""
    equipment_description: str = ""
    logo_description: str = ""

    # ── Environment ───────────────────────────────────────────────────────────
    neighborhood_style: str = ""
    driveway_style: str = ""
    vegetation_style: str = ""
    typical_weather: str = ""
    typical_seasons: list[str] = field(default_factory=list)

    # ── Work scenarios ────────────────────────────────────────────────────────
    common_service_scenarios: list[str] = field(default_factory=list)
    garage_door_styles: list[str] = field(default_factory=list)
    typical_repair_contexts: list[str] = field(default_factory=list)

    # ── Photography style ─────────────────────────────────────────────────────
    camera_angle: str = ""
    focal_length_feel: str = ""
    exposure_style: str = ""
    color_temperature: str = ""
    color_grading: str = ""
    depth_of_field: str = ""
    sharpness_notes: str = ""
    imperfections: str = ""
    framing_style: str = ""
    documentary_notes: str = ""

    # ── Prompt construction ───────────────────────────────────────────────────
    forbidden_elements: list[str] = field(default_factory=list)
    identity_summary: str = ""

    # ── Provenance ────────────────────────────────────────────────────────────
    reference_file_ids: list[str] = field(default_factory=list)
    training_image_count: int = 0

    def to_prompt_context(self) -> str:
        """
        Compact text summary for inclusion in AI generation prompts.

        Only populated fields are included. The result is appended to both
        variation prompts and scratch-generation prompts so the AI knows
        exactly what identity to replicate.
        """
        parts: list[str] = []
        if self.identity_summary:
            parts.append(self.identity_summary)
        if self.technician_description:
            parts.append(f"Technician: {self.technician_description}")
        if self.uniform_style:
            parts.append(f"Uniform: {self.uniform_style}")
        if self.truck_description:
            parts.append(f"Truck: {self.truck_description}")
        if self.logo_description:
            parts.append(f"Company logo: {self.logo_description}")
        if self.neighborhood_style:
            parts.append(f"Neighborhood: {self.neighborhood_style}")
        if self.driveway_style:
            parts.append(f"Driveways: {self.driveway_style}")
        if self.typical_weather:
            parts.append(f"Weather/lighting: {self.typical_weather}")
        if self.camera_angle:
            parts.append(f"Camera angle: {self.camera_angle}")
        if self.focal_length_feel:
            parts.append(f"Focal length: {self.focal_length_feel}")
        if self.color_grading:
            parts.append(f"Color grading: {self.color_grading}")
        if self.imperfections:
            parts.append(f"Photo imperfections: {self.imperfections}")
        if self.documentary_notes:
            parts.append(f"Style: {self.documentary_notes}")
        if self.forbidden_elements:
            parts.append("NEVER include: " + "; ".join(self.forbidden_elements))
        return "\n".join(parts)

    def to_qa_context(self) -> str:
        """
        Text summary for vision QA reviewers evaluating identity preservation.

        More detailed than to_prompt_context() — includes all identity elements
        so reviewers can verify each one against the generated image.
        """
        parts: list[str] = []
        if self.identity_summary:
            parts.append(f"Company identity:\n{self.identity_summary}\n")

        if self.technician_description or self.uniform_style:
            parts.append("PEOPLE:")
            if self.technician_description:
                parts.append(f"  Technician: {self.technician_description}")
            if self.uniform_style:
                parts.append(f"  Uniform: {self.uniform_style}")

        if self.truck_description or self.logo_description or self.equipment_description:
            parts.append("EQUIPMENT:")
            if self.truck_description:
                parts.append(f"  Truck: {self.truck_description}")
            if self.logo_description:
                parts.append(f"  Logo/branding: {self.logo_description}")
            if self.equipment_description:
                parts.append(f"  Tools: {self.equipment_description}")

        if self.neighborhood_style or self.driveway_style or self.vegetation_style:
            parts.append("ENVIRONMENT:")
            if self.neighborhood_style:
                parts.append(f"  Neighborhood: {self.neighborhood_style}")
            if self.driveway_style:
                parts.append(f"  Driveways: {self.driveway_style}")
            if self.vegetation_style:
                parts.append(f"  Vegetation: {self.vegetation_style}")

        if self.camera_angle or self.color_grading or self.imperfections:
            parts.append("PHOTOGRAPHY:")
            if self.camera_angle:
                parts.append(f"  Camera angle: {self.camera_angle}")
            if self.focal_length_feel:
                parts.append(f"  Focal length: {self.focal_length_feel}")
            if self.color_grading:
                parts.append(f"  Color grading: {self.color_grading}")
            if self.imperfections:
                parts.append(f"  Imperfections: {self.imperfections}")

        if self.forbidden_elements:
            parts.append("FORBIDDEN in this company's images:")
            for elem in self.forbidden_elements:
                parts.append(f"  • {elem}")

        return "\n".join(parts)
