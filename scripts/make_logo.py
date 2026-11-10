"""Generate the logo SVGs in assets/. Run: python scripts/make_logo.py

Idea: a smooth curve is the original model. The staircase is the same curve stored with fewer
bits. The amber sliver between them is the gap that quantgap measures.
"""

from __future__ import annotations

import math
from pathlib import Path

NAVY = "#0f172a"
SMOOTH = "#e2e8f0"
STEP = "#3b82f6"
GAP = "#f59e0b"

X0, X1 = 44.0, 212.0  # curve spans x in the 256 box
Y_LOW, Y_HIGH = 206.0, 50.0
LEVELS = 5


def f(t: float) -> float:
    """Smooth rise from 0 to 1 (smoothstep-ish with a gentle start)."""
    return 0.5 - 0.5 * math.cos(math.pi * t) ** 1 if t < 1 else 1.0


def y_of(v: float) -> float:
    return Y_LOW + (Y_HIGH - Y_LOW) * v


def build_icon() -> str:
    n = 80
    xs = [X0 + (X1 - X0) * i / n for i in range(n + 1)]
    smooth = [(x, y_of(f((x - X0) / (X1 - X0)))) for x in xs]
    # staircase: quantize the value to LEVELS steps, drawn as horizontal runs and risers
    stairs: list[tuple[float, float]] = []
    prev = None
    for x in xs:
        v = f((x - X0) / (X1 - X0))
        q = round(v * (LEVELS - 1)) / (LEVELS - 1)
        if prev is not None and q != prev:
            stairs.append((x, y_of(prev)))
        stairs.append((x, y_of(q)))
        prev = q
    gap_poly = smooth + stairs[::-1]

    def path(points: list[tuple[float, float]]) -> str:
        return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)

    def poly(points: list[tuple[float, float]]) -> str:
        return path(points) + " Z"

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256" role="img" aria-label="quantgap logo">
  <title>quantgap</title>
  <rect width="256" height="256" rx="56" fill="{NAVY}"/>
  <path d="{path(stairs)}" fill="none" stroke="{STEP}" stroke-width="8" stroke-linejoin="round" stroke-linecap="round"/>
  <path d="{poly(gap_poly)}" fill="{GAP}"/>
  <path d="{path(smooth)}" fill="none" stroke="{SMOOTH}" stroke-width="5" stroke-linecap="round"/>
</svg>
"""


def build_banner(icon_svg: str) -> str:
    inner = icon_svg.split(">", 1)[1].rsplit("</svg>", 1)[0]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 200" width="760" height="200" role="img" aria-label="quantgap: how much did this quant lose?">
  <title>quantgap: how much did this quant lose?</title>
  <g transform="translate(20,22) scale(0.625)">{inner}</g>
  <text x="200" y="104" font-family="'Segoe UI', system-ui, -apple-system, Helvetica, Arial, sans-serif" font-size="68" font-weight="700" fill="#3b82f6">quant<tspan fill="#f59e0b">gap</tspan></text>
  <text x="203" y="144" font-family="'Segoe UI', system-ui, -apple-system, Helvetica, Arial, sans-serif" font-size="23" fill="#64748b">how much did this quant lose?</text>
</svg>
"""


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "assets"
    out.mkdir(exist_ok=True)
    icon = build_icon()
    (out / "logo.svg").write_text(icon, encoding="utf-8")
    (out / "logo-banner.svg").write_text(build_banner(icon), encoding="utf-8")
    print("wrote", out / "logo.svg", "and", out / "logo-banner.svg")


if __name__ == "__main__":
    main()
