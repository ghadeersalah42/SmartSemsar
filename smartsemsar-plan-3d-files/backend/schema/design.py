"""
Smart Semsar - Design preferences (what the user wants from the interior)
طلبات المستخدم في الفرش: مصدر واحد سواء جت من فورم، كلام/صوت (LLM)، أو صورة (VLM).

Rules:
  - Every input channel fills this same object; the placer reads nothing else.
  - The LLM / VLM only fills these fields. It never decides where furniture goes.
  - Closed vocabularies: style and density are fixed lists; must_have / exclude use the
    catalog's names (Catalog.vocabulary()). Anything else is dropped, never guessed.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

Style = Literal["modern", "classic", "minimal", "industrial", "unknown"]

# minimal = essentials only (bed, sofa, tables); normal = the usual room;
# full = every optional piece that fits, whatever the room size
Density = Literal["minimal", "normal", "full"]


class DesignPreferences(BaseModel):
    style: Style = "unknown"
    # الوصف المحسّن: the LLM's rewrite of the user's own words into one clear style description.
    # Used as the prompt for image models and shown back to the user; the placer ignores it.
    style_brief: Optional[str] = None
    palette: list[str] = Field(default_factory=list)        # main colours, e.g. ["beige", "dark green"]
    materials: list[str] = Field(default_factory=list)      # e.g. ["light wood", "leather"]
    density: Density = "normal"
    must_have: list[str] = Field(default_factory=list)      # catalog id / kind / category to force
    exclude: list[str] = Field(default_factory=list)        # catalog id / kind / category to leave out
    dining_seats: Optional[Literal[4, 6]] = None            # None = the placer picks from the room size
    notes: Optional[str] = None
    missing_fields: list[str] = Field(default_factory=list)  # fields the user did not state or that were unusable

    @classmethod
    def from_loose(cls, data: dict[str, Any]) -> "DesignPreferences":
        """Build from LLM / VLM output. A field with an unusable value ("style": "cozy")
        falls back to its default and is listed in missing_fields instead of failing."""
        data = dict(data or {})
        dropped: list[str] = []
        while True:
            try:
                prefs = cls.model_validate(data)
                break
            except ValidationError as e:
                bad = {str(err["loc"][0]) for err in e.errors() if err["loc"]}
                if not bad & data.keys():
                    raise
                for name in sorted(bad & data.keys()):
                    data.pop(name)
                    dropped.append(name)
        prefs.missing_fields = sorted(set(prefs.missing_fields) | set(dropped))
        return prefs

    def unknown_names(self, vocabulary: set[str]) -> list[str]:
        """must_have / exclude entries that are not catalog names."""
        return sorted({n for n in self.must_have + self.exclude if n not in vocabulary})


# ---------- what the vision model saw in an uploaded photo ----------
PhotoScene = Literal["single_item", "several_items", "room", "not_furniture"]

NOT_USABLE = {
    "room": "This photo shows a whole room. Upload a photo of one piece of furniture; "
            "the room's style was still read from it.",
    "several_items": "This photo shows several pieces. Upload one piece at a time.",
    "not_furniture": "No furniture was found in this photo.",
}


class FurniturePhoto(BaseModel):
    """Result of the VLM tool. `usable` is decided here from checked fields, never by the model."""
    usable: bool = False                    # one whole piece we have a place for -> send to image-to-3D
    reason: Optional[str] = None            # why not usable, in words for the user
    scene: PhotoScene = "not_furniture"
    kind: Optional[str] = None              # a catalog kind, or None
    name: Optional[str] = None              # e.g. "round swivel armchair"
    width_m: Optional[float] = None         # estimate of the real width; None if missing or absurd
    style: Style = "unknown"
    colors: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    description: Optional[str] = None

    @classmethod
    def from_model_answer(cls, data: dict[str, Any], kinds: list[str],
                          width_range_m: tuple[float, float] = (0.3, 3.5)) -> "FurniturePhoto":
        """Check every field of the model's JSON; anything unusable becomes the safe default."""
        def words(value) -> list[str]:
            return [str(v).strip() for v in value if str(v).strip()][:3] if isinstance(value, list) else []

        def text(value) -> Optional[str]:
            return str(value).strip() or None if isinstance(value, str) else None

        scene = data.get("scene") if data.get("scene") in PhotoScene.__args__ else "not_furniture"
        style = data.get("style") if data.get("style") in Style.__args__ else "unknown"
        kind = data.get("kind") if data.get("kind") in kinds else None
        width = data.get("width_m")
        if isinstance(width, bool) or not isinstance(width, (int, float)) \
                or not width_range_m[0] <= width <= width_range_m[1]:
            width = None
        name = text(data.get("name"))

        if scene != "single_item":
            kind, width, reason = None, None, NOT_USABLE[scene]
        elif kind is None:
            reason = f"We have no place in the plan for this piece ({name or 'unknown'})."
        elif data.get("fully_visible") is False:
            reason = "The piece is cut off or partly hidden. Upload a photo that shows all of it."
        else:
            reason = None
        return cls(usable=reason is None, reason=reason, scene=scene, kind=kind, name=name,
                   width_m=round(float(width), 2) if width is not None else None, style=style,
                   colors=words(data.get("colors")), materials=words(data.get("materials")),
                   description=text(data.get("description")))

    def style_preferences(self) -> dict[str, Any]:
        """The part of the photo that feeds DesignPreferences (works for room photos too)."""
        prefs: dict[str, Any] = {}
        if self.style != "unknown":
            prefs["style"] = self.style
        if self.colors:
            prefs["palette"] = self.colors
        if self.materials:
            prefs["materials"] = self.materials
        if self.description:
            prefs["style_brief"] = self.description
        return prefs
