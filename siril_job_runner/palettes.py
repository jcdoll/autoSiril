"""
Narrowband palette definitions for SHO/HOO composition.

Palettes define how narrowband filter channels (S, H, O) map to RGB.
Simple palettes use direct 1:1 mapping (rgbcomp).
Blended palettes use PixelMath expressions for mixing channels.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Palette:
    """Narrowband palette definition with optional channel blending."""

    name: str
    required: set[str]  # Required filter channels (excluding L)
    r: str  # R channel formula
    g: str  # G channel formula
    b: str  # B channel formula
    dynamic: bool = False  # If True, requires non-linear (stretched) input data

    def is_simple(self) -> bool:
        """Check if palette uses simple 1:1 channel mapping (no blending)."""
        return (
            self.r in self.required
            and self.g in self.required
            and self.b in self.required
        )


# Palette registry
# Formulas use channel names directly (e.g., "0.5*H + 0.5*O")
# These get converted to PixelMath syntax at runtime
PALETTES: dict[str, Palette] = {
    # Simple 1:1 mappings
    "HOO": Palette(
        name="HOO",
        required={"H", "O"},
        r="H",
        g="O",
        b="O",
    ),
    "SHO": Palette(
        name="SHO (Hubble)",
        required={"S", "H", "O"},
        r="S",
        g="H",
        b="O",
    ),
    # Blended palettes
    "SHO_FORAXX": Palette(
        name="SHO Foraxx",
        required={"S", "H", "O"},
        r="S",
        g="0.5*H + 0.5*O",
        b="O",
    ),
    "SHO_DYNAMIC": Palette(
        name="SHO Dynamic",
        required={"S", "H", "O"},
        r="0.8*S + 0.2*H",
        g="0.7*H + 0.15*S + 0.15*O",
        b="0.8*O + 0.2*H",
    ),
    # Gold/warm palettes for yellow-blue bicolor look
    "SHO_GOLD": Palette(
        name="SHO Gold",
        required={"S", "H", "O"},
        r="0.8*H + 0.2*S",
        g="0.5*H + 0.5*O",
        b="O",
    ),
    "SHO_WARM": Palette(
        name="SHO Warm",
        required={"S", "H", "O"},
        r="0.75*S + 0.25*H",
        g="H",
        b="O",
    ),
    "SHO_BLUEGOLD": Palette(
        name="SHO Blue-Gold",
        required={"S", "H", "O"},
        r="0.8*H + 0.2*S",
        g="0.7*O + 0.3*H",
        b="O",
    ),
    # Dynamic Foraxx palette - per-pixel adaptive blending
    # Sources:
    #   https://thecoldestnights.com/2020/06/pixinsight-dynamic-narrowband-combinations-with-pixelmath/
    #   https://telescope.live/blog/dynamic-narrowband-combinations-pixelmath
    # Original PixInsight formulas (~ = invert, ^ = power):
    #   R = (Oiii^~Oiii)*Sii + ~(Oiii^~Oiii)*Ha
    #   G = ((Oiii*Ha)^~(Oiii*Ha))*Ha + ~((Oiii*Ha)^~(Oiii*Ha))*Oiii
    #   B = Oiii
    # Note: Formulas use normalized [0,1] values. Siril PixelMath auto-normalizes
    # 16-bit data to [0,1] range, so we use the formulas directly.
    "SHO_FORAXX_DYNAMIC": Palette(
        name="SHO Foraxx Dynamic",
        required={"S", "H", "O"},
        r="(O^(1-O))*S + (1-(O^(1-O)))*H",
        g="((O*H)^(1-O*H))*H + (1-((O*H)^(1-O*H)))*O",
        b="O",
        dynamic=True,  # Requires stretched input data
    ),
    # HOO-style dynamic palette for blue/gold colors
    # Based on Foraxx method but simpler R channel (just H, no S)
    # Source: https://thecoldestnights.com/2020/06/pixinsight-dynamic-narrowband-combinations-with-pixelmath/
    # Produces gold (H-dominant) and blue (O-dominant) with dynamic G transition
    "HOO_FORAXX_DYNAMIC": Palette(
        name="HOO Foraxx Dynamic",
        required={"H", "O"},
        r="H",
        g="((O*H)^(1-O*H))*H + (1-((O*H)^(1-O*H)))*O",
        b="O",
        dynamic=True,  # Requires stretched input data
    ),
}


def get_palette(name: str) -> Palette:
    """Get palette by name. Raises KeyError if not found."""
    if name not in PALETTES:
        available = ", ".join(sorted(PALETTES.keys()))
        raise KeyError(f"Unknown palette: {name}. Available: {available}")
    return PALETTES[name]


def formula_to_pixelmath(formula: str) -> str:
    """
    Convert formula to Siril PixelMath syntax.

    Input: "0.5*H + 0.5*O"
    Output: "$H$ * 0.5 + $O$ * 0.5"
    """
    # Find all channel references (single uppercase letters)
    # Replace them with $X$ syntax
    result = formula

    # Match channel names: S, H, O, L (not preceded/followed by alphanumeric)
    for channel in ["S", "H", "O", "L"]:
        # Replace standalone channel references with $channel$ syntax
        # Use word boundaries to avoid replacing inside other tokens
        result = re.sub(rf"\b{channel}\b", f"${channel}$", result)

    # Normalize spacing around operators for readability
    result = re.sub(r"\s*\*\s*", " * ", result)
    result = re.sub(r"\s*\+\s*", " + ", result)
    result = re.sub(r"\s*-\s*", " - ", result)

    return result.strip()


def build_effective_palette(
    base_palette: Palette,
    r_override: Optional[str] = None,
    g_override: Optional[str] = None,
    b_override: Optional[str] = None,
) -> Palette:
    """
    Create palette with optional per-channel overrides.

    Args:
        base_palette: Base palette to start from
        r_override: Optional R channel formula override
        g_override: Optional G channel formula override
        b_override: Optional B channel formula override

    Returns:
        New Palette with overrides applied
    """
    return Palette(
        name=f"{base_palette.name} (custom)"
        if any([r_override, g_override, b_override])
        else base_palette.name,
        required=base_palette.required,
        r=r_override or base_palette.r,
        g=g_override or base_palette.g,
        b=b_override or base_palette.b,
        dynamic=base_palette.dynamic,
    )
