"""Render the launch video: docs/launch.mp4 (+ a small docs/launch.gif preview for the README).

    python scripts/make_video.py                # render everything (needs ffmpeg on PATH)
    python scripts/make_video.py --check        # draw the last frame of each scene, check layout, save PNGs
    python scripts/make_video.py --still 4.2    # save one frame at 4.2 seconds as a PNG

Everything on screen comes from real data: the numbers are read from docs/results/*.json (the
measured results of the sample run) and the recommendation sentence is produced by the real
`recommend()` function. Nothing is invented.

Layout safety: every text item is registered with its bounding box. Each frame is checked so that
no two texts overlap and nothing leaves the safe margins. A failing check stops the render.

Fonts: Segoe UI and Cascadia Mono (Windows). On other systems it falls back to DejaVu.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from functools import cache
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H, S, FPS = 1920, 1080, 2, 30
XFADE = 0.5

BG = (15, 23, 42)
PANEL = (21, 31, 46)
PANEL2 = (30, 41, 59)
TERM = (10, 16, 28)
INK = (226, 232, 240)
MUTED = (148, 163, 184)
BLUE = (59, 130, 246)
BLUE_L = (125, 180, 255)
AMBER = (245, 158, 11)
GREEN = (74, 222, 128)
LINE = (51, 65, 85)

FONT_FILES = {
    "ui": ["segoeui.ttf", "DejaVuSans.ttf"],
    "bold": ["segoeuib.ttf", "DejaVuSans-Bold.ttf"],
    "mono": ["CascadiaMono.ttf", "consola.ttf", "DejaVuSansMono.ttf"],
}
FONT_DIRS = [
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts/Supplemental"),
]


@cache
def font(face: str, size: int) -> ImageFont.FreeTypeFont:
    for name in FONT_FILES[face]:
        for d in FONT_DIRS:
            if (d / name).exists():
                return ImageFont.truetype(str(d / name), int(size * S))
    raise RuntimeError(f"no font found for {face}")


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def ease(x: float) -> float:
    x = clamp(x)
    return 1 - (1 - x) ** 3


def mix(a, b, t):
    t = clamp(t)
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def appear(t: float, t0: float, dur: float = 0.5) -> float:
    return ease((t - t0) / dur)


# ----------------------------------------------------------------------------------------- canvas


class Canvas:
    def __init__(self) -> None:
        self.img = Image.new("RGB", (W * S, H * S), BG)
        self.d = ImageDraw.Draw(self.img)
        self.items: list[tuple[str, tuple[float, float, float, float]]] = []

    # --- text
    def text(self, x, y, s, size, fill=INK, face="ui", anchor="la", a=1.0, bg=BG, tag=None, dy=0.0):
        if a <= 0.02 or not s:
            return None
        f = font(face, size)
        y = y + dy
        self.d.text((x * S, y * S), s, font=f, fill=mix(bg, fill, a), anchor=anchor)
        bb = self.d.textbbox((x * S, y * S), s, font=f, anchor=anchor)
        box = tuple(v / S for v in bb)
        self.items.append((tag or s[:24], box))
        return box

    def runs(self, cx, y, parts, size, face="ui", a=1.0, bg=BG, tag=None, dy=0.0):
        """Centered line made of differently coloured pieces: parts = [(text, colour), ...]."""
        f = font(face, size)
        widths = [self.d.textlength(p, font=f) / S for p, _ in parts]
        x = cx - sum(widths) / 2
        for (p, col), w in zip(parts, widths, strict=True):
            self.text(x, y, p, size, col, face, "la", a, bg, tag or f"run{y}", dy)
            x += w

    def wrap(self, s, size, width, face="ui"):
        f = font(face, size)
        lines, cur = [], ""
        for word in s.split():
            trial = (cur + " " + word).strip()
            if self.d.textlength(trial, font=f) / S <= width or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines

    def para(
        self, x, y, s, size, width, fill=INK, face="ui", a=1.0, bg=BG, lh=1.35, tag=None, dy=0.0
    ):
        """Wrapped paragraph. Returns the y below the last line."""
        for i, line in enumerate(self.wrap(s, size, width, face)):
            self.text(
                x,
                y + i * size * lh,
                line,
                size,
                fill,
                face,
                "la",
                a,
                bg,
                f"{tag or s[:12]}#{i}",
                dy,
            )
        return y + len(self.wrap(s, size, width, face)) * size * lh

    # --- shapes
    def rrect(self, box, r, fill=None, outline=None, width=2, a=1.0, bg=BG):
        x0, y0, x1, y1 = (v * S for v in box)
        self.d.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=r * S,
            fill=mix(bg, fill, a) if fill else None,
            outline=mix(bg, outline, a) if outline else None,
            width=int(width * S),
        )

    def line(self, pts, fill, width=2, a=1.0, bg=BG):
        self.d.line(
            [(x * S, y * S) for x, y in pts],
            fill=mix(bg, fill, a),
            width=int(width * S),
            joint="curve",
        )

    def circle(self, cx, cy, r, fill=None, outline=None, width=2, a=1.0, bg=BG):
        self.d.ellipse(
            ((cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S),
            fill=mix(bg, fill, a) if fill else None,
            outline=mix(bg, outline, a) if outline else None,
            width=int(width * S),
        )

    def polygon(self, pts, fill, a=1.0, bg=BG):
        self.d.polygon([(x * S, y * S) for x, y in pts], fill=mix(bg, fill, a))

    # --- safety
    def check(self, where: str) -> None:
        m_x, m_y = 40, 24
        for tag, (x0, y0, x1, y1) in self.items:
            if x0 < m_x or x1 > W - m_x or y0 < m_y or y1 > H - m_y:
                raise AssertionError(
                    f"{where}: '{tag}' leaves the safe area: {x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}"
                )
        for i in range(len(self.items)):
            for j in range(i + 1, len(self.items)):
                (ta, a), (tb, b) = self.items[i], self.items[j]
                if ta == tb:  # pieces of one line (see runs())
                    continue
                gap = 2
                if (
                    a[0] < b[2] + gap
                    and b[0] < a[2] + gap
                    and a[1] < b[3] + gap
                    and b[1] < a[3] + gap
                ):
                    raise AssertionError(f"{where}: text overlap between '{ta}' and '{tb}'")

    def inside(self, where, box, tags_from=0):
        x0, y0, x1, y1 = box
        for tag, (a0, b0, a1, b1) in self.items[tags_from:]:
            if a0 < x0 or b0 < y0 or a1 > x1 or b1 > y1:
                raise AssertionError(f"{where}: '{tag}' sticks out of its card")

    def frame(self) -> Image.Image:
        return self.img.resize((W, H), Image.LANCZOS)


# ----------------------------------------------------------------------------------------- logo


def draw_logo(c: Canvas, cx, cy, size, a=1.0):
    """Same geometry as scripts/make_logo.py, drawn with the canvas."""
    k = size / 256
    ox, oy = cx - size / 2, cy - size / 2

    def P(x, y):
        return ox + x * k, oy + y * k

    c.rrect((ox, oy, ox + size, oy + size), 56 * k, fill=(15, 23, 42), a=a, bg=BG)
    c.rrect((ox, oy, ox + size, oy + size), 56 * k, outline=LINE, width=2, a=a * 0.8, bg=BG)
    x0, x1, y_low, y_high, levels = 44.0, 212.0, 206.0, 50.0, 5

    def f(t):
        return 0.5 - 0.5 * math.cos(math.pi * t)

    n = 80
    xs = [x0 + (x1 - x0) * i / n for i in range(n + 1)]
    yv = lambda v: y_low + (y_high - y_low) * v  # noqa: E731
    smooth = [(x, yv(f((x - x0) / (x1 - x0)))) for x in xs]
    stairs, prev = [], None
    for x in xs:
        q = round(f((x - x0) / (x1 - x0)) * (levels - 1)) / (levels - 1)
        if prev is not None and q != prev:
            stairs.append((x, yv(prev)))
        stairs.append((x, yv(q)))
        prev = q
    c.line([P(*p) for p in stairs], BLUE, width=8 * k, a=a, bg=(15, 23, 42))
    c.polygon([P(*p) for p in smooth + stairs[::-1]], AMBER, a=a, bg=(15, 23, 42))
    c.line([P(*p) for p in smooth], (226, 232, 240), width=5 * k, a=a, bg=(15, 23, 42))


# ----------------------------------------------------------------------------------------- data


def load_data():
    sys.path.insert(0, str(ROOT / "src"))
    from quantgap.hub.submit import load_result
    from quantgap.score.recommend import ASSUMPTION, recommend
    from quantgap.util import human_size, parse_size

    results = [load_result(f) for f in sorted((ROOT / "docs" / "results").glob("*.json"))]
    return results, recommend, ASSUMPTION, human_size, parse_size


RESULTS, _recommend, ASSUMPTION, human_size, parse_size = load_data()
BY = {(r.quant.publisher, r.quant.label): r for r in RESULTS}


def result(pub, label):
    return BY[(pub, label)]


PACK_NAME = RESULTS[0].pack_name


def score_lines(r):
    m, d = r.metrics, r.metrics.damage
    return [
        ("hdr", f"quantgap {r.quantgap}  pack: {r.pack_name}"),
        (
            "hdr",
            f"quant: {r.quant.label}  {human_size(r.quant.size_bytes)}  {r.backend.name}",
        ),
        ("gap", ""),
        ("col", ("metric", "value", "note")),
        (
            "row",
            (
                "KLD mean",
                f"{m.kld_mean:.4f}",
                f"95% CI [{m.kld_mean_ci95[0]:.4f}, {m.kld_mean_ci95[1]:.4f}]  (lower is better)",
            ),
        ),
        ("row", ("KLD worst 1%", f"{m.kld_p99:.3f}", "99th percentile of positions")),
        ("row", ("top-1 agree", f"{m.top1_agree * 100:.1f} %", "same best guess as the original")),
        ("row", ("ppl ratio", f"{m.ppl_ratio:.4f}", "1.0 means no change")),
        ("row", ("loops", f"{d.loops} / {d.n_prompts}", f"original: {d.loops_ref}")),
        (
            "row",
            (
                "wrong script",
                f"{d.wrong_script} / {d.n_prompts}",
                "script not in the original's text",
            ),
        ),
        ("row", ("valid JSON", f"{d.json_valid} / {d.json_n}", f"original: {d.json_valid_ref}")),
    ]


def compare_view():
    rs = [r for r in RESULTS if r.quant.publisher == "bartowski" and r.quant.label.endswith("K_M")]
    rs.sort(key=lambda r: r.metrics.kld_mean)
    rec = _recommend(rs, parse_size("1.6GB"))
    head = f"{'quant':<9} {'size':<10} {'KLD mean':<10} {'95% CI':<15} {'top-1':<7} ppl"
    rows = [
        f"{r.quant.label:<9} {human_size(r.quant.size_bytes):<10} {r.metrics.kld_mean:<10.4f} "
        f"{r.metrics.kld_mean_ci95[0]:.4f}-{r.metrics.kld_mean_ci95[1]:.4f}  {r.metrics.top1_agree * 100:>4.1f}%  {r.metrics.ppl_ratio:.3f}"
        for r in rs
    ]
    return head, rows, rec.sentence


# ----------------------------------------------------------------------------------------- scenes
# Each scene is draw(c, t) with t in seconds from the scene start. DUR is its length.


def kicker_title(c, t, kicker, title, t0=0.0):
    a = appear(t, t0)
    c.text(110, 78, kicker, 26, AMBER, "bold", a=a, dy=(1 - a) * 12, tag="kicker")
    c.text(
        110,
        118,
        title,
        62,
        INK,
        "bold",
        a=appear(t, t0 + 0.1),
        dy=(1 - appear(t, t0 + 0.1)) * 14,
        tag="title",
    )


def watermark(c, t):
    c.text(W - 110, 1012, "quantgap", 22, MUTED, "bold", anchor="ra", a=0.8, tag="wm")


def scene_title(c, t):
    draw_logo(c, 960, 300, 250, a=appear(t, 0.2, 0.8))
    a = appear(t, 0.8)
    c.runs(
        960, 470, [("quant", BLUE_L), ("gap", AMBER)], 132, "bold", a=a, dy=(1 - a) * 16, tag="word"
    )
    c.text(
        960,
        650,
        "How much did this quant lose?",
        56,
        MUTED,
        "ui",
        anchor="ma",
        a=appear(t, 1.6),
        dy=(1 - appear(t, 1.6)) * 14,
        tag="tag",
    )
    c.text(
        960,
        745,
        "Measure how far a quantized LLM drifts from the original,",
        34,
        INK,
        "ui",
        anchor="ma",
        a=appear(t, 2.4),
        tag="s1",
    )
    c.text(
        960,
        792,
        "on your own machine, without running the original.",
        34,
        INK,
        "ui",
        anchor="ma",
        a=appear(t, 2.6),
        tag="s2",
    )


def scene_problem(c, t):
    a = appear(t, 0.2)
    c.text(
        960,
        190,
        "Which quant should I download?",
        88,
        INK,
        "bold",
        anchor="ma",
        a=a,
        dy=(1 - a) * 18,
        tag="q",
    )
    chips = ["Q8_0", "Q6_K", "Q4_K_M", "UD-Q4_K_XL", "IQ3_XXS", "NVFP4"]
    f = font("mono", 38)
    ws = [c.d.textlength(s, font=f) / S + 64 for s in chips]
    gap = 26
    x = 960 - (sum(ws) + gap * (len(ws) - 1)) / 2
    for i, (s, w) in enumerate(zip(chips, ws, strict=True)):
        ca = appear(t, 1.0 + i * 0.22, 0.4)
        c.rrect((x, 400, x + w, 478), 39, fill=PANEL2, outline=LINE, a=ca)
        c.text(x + w / 2, 439, s, 38, BLUE_L, "mono", anchor="mm", a=ca, bg=PANEL2, tag=f"chip{i}")
        x += w + gap
    a2 = appear(t, 3.4)
    c.text(
        960,
        590,
        "Everyone answers with a feeling.",
        50,
        MUTED,
        "ui",
        anchor="ma",
        a=a2,
        dy=(1 - a2) * 14,
        tag="feel",
    )
    a3 = appear(t, 5.2)
    c.runs(
        960,
        700,
        [("quantgap answers with a ", INK), ("number.", AMBER)],
        62,
        "bold",
        a=a3,
        dy=(1 - a3) * 14,
        tag="num",
    )


def scene_idea(c, t):
    kicker_title(c, t, "HOW IT WORKS", "The big model runs once. Everyone reuses it.")
    cards = [
        (
            "1",
            "Pack, once",
            "Someone runs the original model over 160 fixed texts and saves its top 32 guesses at every position. A small shared file: about 10 MB for this 0.5B model.",
        ),
        (
            "2",
            "Score, anywhere",
            "You run your quant on your own laptop over the same texts. llama.cpp, Transformers, or any vLLM server.",
        ),
        (
            "3",
            "See the gap",
            "One number says how far the quant drifted from the original. Lower is better. Plus checks for loops, stray scripts and broken JSON.",
        ),
    ]
    cw, ch, gap, x0, y0 = 520, 420, 80, 100, 280
    for i, (n, title, body) in enumerate(cards):
        a = appear(t, 1.0 + i * 1.6, 0.6)
        x = x0 + i * (cw + gap)
        dy = (1 - a) * 22
        top = y0 + dy
        c.rrect((x, top, x + cw, top + ch), 28, fill=PANEL, outline=LINE, a=a)
        c.circle(x + 66, top + 70, 34, fill=BLUE, a=a, bg=PANEL)
        c.text(
            x + 66,
            top + 70,
            n,
            36,
            (255, 255, 255),
            "bold",
            anchor="mm",
            a=a,
            bg=PANEL,
            tag=f"n{n}",
        )
        c.text(x + 120, top + 70, title, 44, INK, "bold", anchor="lm", a=a, bg=PANEL, tag=f"t{n}")
        c.para(x + 40, top + 140, body, 29, cw - 80, MUTED, a=a, bg=PANEL, lh=1.4, tag=f"b{n}")
        if i < 2:
            ax = x + cw + gap / 2
            aa = appear(t, 1.0 + (i + 1) * 1.6 - 0.3, 0.4)
            c.line(
                [(ax - 14, y0 + ch / 2 - 22), (ax + 10, y0 + ch / 2), (ax - 14, y0 + ch / 2 + 22)],
                AMBER,
                width=6,
                a=aa,
            )
    a = appear(t, 6.4)
    c.text(
        960,
        790,
        "No GPU needed to score. No need to ever run the original.",
        40,
        INK,
        "ui",
        anchor="ma",
        a=a,
        dy=(1 - a) * 12,
        tag="foot",
    )


class Terminal:
    """A typing terminal. Lines are (kind, text); commands are typed, output lines pop in."""

    X0, Y0, X1, Y1 = 110, 240, 1810, 985

    def __init__(self, c: Canvas, size=26):
        self.c, self.size = c, size
        self.cw = c.d.textlength("M", font=font("mono", size)) / S
        self.lh = size * 1.62

    def window(self, t, title):
        c = self.c
        a = appear(t, 0.1, 0.5)
        c.rrect((self.X0, self.Y0, self.X1, self.Y1), 22, fill=TERM, outline=LINE, a=a)
        c.rrect((self.X0, self.Y0, self.X1, self.Y0 + 56), 22, fill=PANEL, a=a)
        c.rrect((self.X0, self.Y0 + 30, self.X1, self.Y0 + 56), 0, fill=PANEL, a=a)
        for i, col in enumerate(((248, 113, 113), (251, 191, 36), (74, 222, 128))):
            c.circle(self.X0 + 36 + i * 32, self.Y0 + 28, 9, fill=col, a=a)
        c.text(
            (self.X0 + self.X1) / 2,
            self.Y0 + 28,
            title,
            22,
            MUTED,
            "ui",
            anchor="mm",
            a=a,
            bg=PANEL,
            tag="wtitle",
        )

    def cmd(self, row, text, t, start, cps, cursor_end=False):
        """Type a command on grid row `row`. Returns the time it finishes."""
        c = self.c
        n = int(clamp((t - start) * cps, 0, len(text)))
        x, y = self.X0 + 40, self.Y0 + 86 + row * self.lh
        if t >= start:
            c.text(x, y, "$", self.size, GREEN, "mono", bg=TERM, tag=f"p{row}")
            shown = text[:n]
            if shown:
                parts = shown.split(" ")
                cx = x + 2 * self.cw
                for k, w in enumerate(parts):
                    colour = BLUE_L if w.startswith(("hf://", "--", "https://")) else INK
                    if k == 0 and w == "quantgap":
                        colour = AMBER
                    c.text(cx, y, w, self.size, colour, "mono", bg=TERM, tag=f"c{row}.{k}")
                    cx += (len(w) + 1) * self.cw
            done = n >= len(text)
            blink = int(t * 2.4) % 2 == 0
            if (not done) or (cursor_end and blink):
                cxp = x + (2 + n) * self.cw
                c.rrect(
                    (cxp, y + 3, cxp + self.cw * 0.75, y + self.size * 1.25),
                    2,
                    fill=INK,
                    a=0.85,
                    bg=TERM,
                )
        return start + len(text) / cps

    def out(self, row, parts, t, start):
        """One output line made of (column, text, colour) at fixed character columns."""
        a = appear(t, start, 0.25)
        if a <= 0.02:
            return
        x, y = self.X0 + 40, self.Y0 + 86 + row * self.lh
        for col, s, colour in parts:
            if s:
                self.c.text(
                    x + col * self.cw,
                    y,
                    s,
                    self.size,
                    colour,
                    "mono",
                    a=a,
                    bg=TERM,
                    dy=(1 - a) * 6,
                    tag=f"o{row}:{col}",
                )


SCORE_CMD_URL = "hf://bartowski/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"


def scene_score(c, t):
    kicker_title(c, t, "USE IT", "1. Install, then score one quant")
    term = Terminal(c, 26)
    term.window(t, "terminal")
    e1 = term.cmd(
        0,
        'pip install "quantgap[llamacpp]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu',
        t,
        0.9,
        60,
    )
    cmd2 = f"quantgap score {SCORE_CMD_URL}"
    e2 = term.cmd(1, cmd2, t, e1 + 0.7, 62, cursor_end=False)
    lines = score_lines(result("bartowski", "Q4_K_M"))
    row, start = 2, e2 + 0.7
    for kind, val in lines:
        if kind == "hdr":
            term.out(row, [(0, val, INK)], t, start)
        elif kind == "col":
            term.out(row, [(0, val[0], MUTED), (18, val[1], MUTED), (27, val[2], MUTED)], t, start)
        elif kind == "row":
            term.out(row, [(0, val[0], INK), (18, val[1], BLUE_L), (27, val[2], MUTED)], t, start)
        row += 1
        start += 0.22
    a = appear(t, start + 0.4)
    c.text(
        110,
        1020,
        "Real output: Qwen2.5-0.5B-Instruct Q4_K_M, full mode, free Colab T4 GPU.",
        22,
        MUTED,
        a=a,
        tag="cap",
    )


def scene_compare(c, t):
    kicker_title(c, t, "USE IT", "2. Compare quants and pick one for your memory")
    term = Terminal(c, 26)
    term.window(t, "terminal")
    cmd = (
        'quantgap compare --repo bartowski/Qwen2.5-0.5B-Instruct-GGUF --glob "*K_M*" --budget 1.6GB'
    )
    e = term.cmd(0, cmd, t, 0.9, 62)
    head, rows, sentence = compare_view()
    start = e + 0.8
    term.out(2, [(0, head, MUTED)], t, start)
    for i, r in enumerate(rows):
        term.out(3 + i, [(0, r, INK)], t, start + 0.3 + i * 0.3)
    s2 = start + 0.3 + len(rows) * 0.3 + 0.5
    term.out(3 + len(rows) + 1, [(0, sentence, AMBER)], t, s2 + 0.2)
    term.out(
        3 + len(rows) + 2,
        [(0, "Memory estimate = file size x 1.1 + 1 GiB (context buffers at 2K context).", MUTED)],
        t,
        s2 + 0.6,
    )
    a = appear(t, s2 + 1.4)
    c.text(
        110,
        1020,
        "Real numbers from the full-mode sample run (columns trimmed to fit). Add --chart chart.svg to also draw the quality-vs-size chart.",
        22,
        MUTED,
        a=a,
        tag="cap",
    )


def scene_chart(c, t):
    kicker_title(c, t, "REAL RESULTS", "Smaller files drift further from the original")
    px0, py0, px1, py1 = 110, 215, 1350, 985
    c.rrect((px0, py0, px1, py1), 24, fill=PANEL, outline=LINE)
    L, R, T, B = 215, 1310, 300, 880
    kmin, kmax = 0.002, 0.3
    smin, smax = 320.0, 560.0

    def X(mb):
        return L + (mb - smin) / (smax - smin) * (R - L)

    def Y(k):
        return B - (math.log10(k) - math.log10(kmin)) / (math.log10(kmax) - math.log10(kmin)) * (
            B - T
        )

    c.text(
        px0 + 36,
        py0 + 24,
        "KLD mean vs the original (log scale, lower is better)",
        24,
        MUTED,
        bg=PANEL,
        tag="ytitle",
    )
    for k in (0.003, 0.01, 0.03, 0.1):
        c.line([(L, Y(k)), (R, Y(k))], LINE, 1, bg=PANEL)
        c.text(L - 14, Y(k), f"{k:g}", 22, MUTED, anchor="rm", bg=PANEL, tag=f"yt{k}")
    for mb in (350, 400, 450, 500, 550):
        c.text(X(mb), B + 22, f"{mb}", 22, MUTED, anchor="ma", bg=PANEL, tag=f"xt{mb}")
    c.text((L + R) / 2, B + 62, "File size (MB)", 24, MUTED, anchor="ma", bg=PANEL, tag="xtitle")

    order = [("bartowski", "Q2_K"), ("bartowski", "Q3_K_M"), ("bartowski", "Q4_K_M"), ("bartowski", "Q5_K_M"),
             ("Qwen", "Q4_K_M"), ("bartowski", "Q6_K"), ("bartowski", "Q8_0")]  # fmt: skip
    # label placement: (dx, dy, anchor)
    place = {("bartowski", "Q2_K"): (16, 0, "lm"), ("bartowski", "Q3_K_M"): (16, 0, "lm"),
             ("bartowski", "Q4_K_M"): (16, 0, "lm"), ("bartowski", "Q5_K_M"): (16, 0, "lm"),
             ("Qwen", "Q4_K_M"): (16, 0, "lm"), ("bartowski", "Q6_K"): (-16, -2, "rm"),
             ("bartowski", "Q8_0"): (-16, -2, "rm")}  # fmt: skip
    pts = []
    for key in order:
        r = result(*key)
        pts.append((key, r.quant.size_bytes / 1e6, r.metrics.kld_mean, r.metrics.kld_mean_ci95))
    # frontier line through the points not beaten by a smaller, better file
    front, best = [], 9.0
    for _key, mb, k, _ in sorted(pts, key=lambda p: p[1]):
        if k < best:
            front.append((X(mb), Y(k)))
            best = k
    la = appear(t, 1.0, 1.2)
    if len(front) > 1:
        c.line(front, BLUE, 2, a=0.45 * la, bg=PANEL)
    for i, (key, mb, k, (lo, hi)) in enumerate(pts):
        a = appear(t, 0.9 + i * 0.5, 0.4)
        x, y = X(mb), Y(k)
        odd = key == ("Qwen", "Q4_K_M")
        col = AMBER if odd else BLUE
        c.line([(x, Y(lo)), (x, Y(hi))], col, 2, a=0.55 * a, bg=PANEL)
        c.circle(x, y, 9, fill=col, outline=PANEL, width=3, a=a, bg=PANEL)
        dx, dy, anc = place[key]
        label = key[1] + (
            " (Qwen)"
            if key == ("Qwen", "Q4_K_M")
            else " (bartowski)"
            if key == ("bartowski", "Q4_K_M")
            else ""
        )
        c.text(
            x + dx,
            y + dy,
            label,
            24,
            INK if not odd else AMBER,
            "bold" if odd else "ui",
            anchor=anc,
            a=a,
            bg=PANEL,
            tag=f"lab{i}",
        )
    # callouts
    q2 = result("bartowski", "Q2_K").metrics.kld_mean
    q8 = result("bartowski", "Q8_0").metrics.kld_mean
    qw = result("Qwen", "Q4_K_M").metrics.kld_mean
    qb = result("bartowski", "Q4_K_M").metrics.kld_mean
    notes = [
        (BLUE_L, "Big files stay close", "Q8_0 and Q6_K barely move from the original."),
        (AMBER, "Small files pay", f"Q2_K drifted about {q2 / q8:.0f} times as far as Q8_0."),
        (
            AMBER,
            "Same label, different file",
            f"Qwen's own Q4_K_M drifted about {qw / qb:.0f} times as far as bartowski's Q4_K_M.",
        ),
    ]
    y = 250
    for i, (col, head, body) in enumerate(notes):
        a = appear(t, 5.0 + i * 1.6, 0.5)
        c.text(1400, y, head, 32, col, "bold", a=a, dy=(1 - a) * 12, tag=f"nh{i}")
        yb = c.para(1400, y + 46, body, 27, 410, MUTED, a=a, lh=1.38, tag=f"nb{i}", dy=(1 - a) * 12)
        y = yb + 34
    a = appear(t, 8.8)
    c.text(1400, 940, "Qwen2.5-0.5B-Instruct, 7 of 15 GGUF files", 21, MUTED, a=a, tag="f1")
    c.text(1400, 970, "shown, full mode, free Colab T4.", 21, MUTED, a=a, tag="f2")


def scene_tiles(c, t):
    kicker_title(c, t, "WHAT YOU GET", "Four numbers, in plain words")
    tiles = [
        (
            "KLD mean",
            "How far the quant's probabilities moved from the original. 0 means identical. Lower is better, with a 95% interval.",
        ),
        (
            "Worst 1%",
            "An average can hide rare big failures. The 99th percentile of drift catches them.",
        ),
        (
            "Top-1 agreement",
            "How often the quant's best next-token guess is the same as the original's.",
        ),
        (
            "Damage checks",
            "Loops, a stray Chinese or Arabic character in English text, JSON that no longer parses. Always shown next to the original's own count.",
        ),
    ]
    cw, ch = 840, 300
    pos = [(110, 260), (970, 260), (110, 600), (970, 600)]
    for i, ((title, body), (x, y)) in enumerate(zip(tiles, pos, strict=True)):
        a = appear(t, 0.7 + i * 0.9, 0.6)
        y2 = y + (1 - a) * 20
        c.rrect((x, y2, x + cw, y2 + ch), 26, fill=PANEL, outline=LINE, a=a)
        c.rrect(
            (x + 34, y2 + 40, x + 44, y2 + 90), 5, fill=AMBER if i == 3 else BLUE, a=a, bg=PANEL
        )
        start = len(c.items)
        c.text(x + 66, y2 + 64, title, 42, INK, "bold", anchor="lm", a=a, bg=PANEL, tag=f"tt{i}")
        c.para(x + 40, y2 + 125, body, 29, cw - 80, MUTED, a=a, bg=PANEL, lh=1.4, tag=f"tb{i}")
        c.inside(f"tile{i}", (x, y2, x + cw, y2 + ch), start)


def scene_share(c, t):
    kicker_title(c, t, "MAKE IT YOURS", "Make a pack. Share a result.")
    cards = [
        (
            "Make a pack for any model",
            "quantgap pack <org>/<model> --revision <sha>",
            "Runs the original once and saves a small shareable pack. A free Colab GPU notebook is included. It suits models up to about 3B parameters.",
        ),
        (
            "Share a result",
            "quantgap submit result.json",
            "Opens a pull request on the results dataset. Every result carries the hash of the quant file and the pack, so anyone can re-run it and check.",
        ),
    ]
    cw, ch, y0 = 840, 470, 280
    for i, (title, cmd, body) in enumerate(cards):
        a = appear(t, 0.7 + i * 1.2, 0.6)
        x = 110 + i * 860
        y = y0 + (1 - a) * 20
        c.rrect((x, y, x + cw, y + ch), 26, fill=PANEL, outline=LINE, a=a)
        start = len(c.items)
        c.text(x + 40, y + 40, title, 42, INK, "bold", a=a, bg=PANEL, tag=f"st{i}")
        c.rrect((x + 40, y + 130, x + cw - 40, y + 210), 16, fill=TERM, outline=LINE, a=a, bg=PANEL)
        c.text(x + 66, y + 170, "$", 24, GREEN, "mono", anchor="lm", a=a, bg=TERM, tag=f"sp{i}")
        c.text(x + 96, y + 170, cmd, 24, INK, "mono", anchor="lm", a=a, bg=TERM, tag=f"sc{i}")
        c.para(x + 40, y + 250, body, 30, cw - 80, MUTED, a=a, bg=PANEL, lh=1.42, tag=f"sb{i}")
        c.inside(f"share{i}", (x + 20, y, x + cw - 20, y + ch), start)


def scene_limits(c, t):
    kicker_title(c, t, "BE HONEST", "What quantgap does not tell you")
    items = [
        (
            "It is not a skill benchmark.",
            "It measures drift from the original, not how good a model is at your task.",
        ),
        (
            "The KLD is a lower bound.",
            "Great for ranking quants of one model. Compare quantgap numbers with quantgap numbers.",
        ),
        (
            "The numbers in this video are a small demo.",
            "One 0.5B model, full mode, one free GPU. Packs for more models are the next step.",
        ),
    ]
    y = 330
    for i, (head, body) in enumerate(items):
        a = appear(t, 0.7 + i * 1.1, 0.6)
        c.circle(130, y + 26, 11, fill=AMBER, a=a)
        c.text(170, y, head, 42, INK, "bold", a=a, dy=(1 - a) * 14, tag=f"lh{i}")
        yb = c.para(170, y + 62, body, 31, 1500, MUTED, a=a, lh=1.4, tag=f"lb{i}", dy=(1 - a) * 14)
        y = yb + 56


def scene_end(c, t):
    draw_logo(c, 960, 230, 190, a=appear(t, 0.1, 0.7))
    a = appear(t, 0.5)
    c.runs(
        960, 360, [("quant", BLUE_L), ("gap", AMBER)], 108, "bold", a=a, dy=(1 - a) * 14, tag="w"
    )
    c.text(
        960,
        510,
        "How much did this quant lose?",
        44,
        MUTED,
        anchor="ma",
        a=appear(t, 1.0),
        tag="tg",
    )
    a = appear(t, 1.6, 0.6)
    pill = 'pip install "quantgap[llamacpp]"'
    wpx = c.d.textlength(pill, font=font("mono", 40)) / S + 90
    c.rrect((960 - wpx / 2, 600, 960 + wpx / 2, 690), 45, fill=PANEL2, outline=BLUE, a=a)
    c.text(960, 645, pill, 40, INK, "mono", anchor="mm", a=a, bg=PANEL2, tag="pill")
    a = appear(t, 2.0)
    c.text(
        960,
        706,
        "No C++ compiler? Add --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu",
        24,
        MUTED,
        anchor="ma",
        a=a,
        tag="idx",
    )
    a = appear(t, 2.6)
    c.runs(
        960,
        775,
        [("GitHub  ", MUTED), ("github.com/saqlain2204/quantgap", BLUE_L)],
        36,
        "ui",
        a=a,
        tag="gh",
    )
    c.runs(
        960,
        830,
        [("Packs  ", MUTED), ("huggingface.co/quantgap-packs", BLUE_L)],
        36,
        "ui",
        a=a,
        tag="pk",
    )
    c.runs(
        960,
        885,
        [("Leaderboard  ", MUTED), ("saqlain2204.github.io/quantgap", BLUE_L)],
        36,
        "ui",
        a=a,
        tag="lb",
    )
    c.text(960, 985, "Open source, Apache-2.0", 28, MUTED, anchor="ma", a=appear(t, 3.2), tag="lic")


SCENES = [
    ("title", scene_title, 5.5, False),
    ("problem", scene_problem, 8.0, True),
    ("idea", scene_idea, 9.5, True),
    ("score", scene_score, 13.5, True),
    ("compare", scene_compare, 12.5, True),
    ("chart", scene_chart, 11.5, True),
    ("tiles", scene_tiles, 8.5, True),
    ("share", scene_share, 7.5, True),
    ("limits", scene_limits, 7.0, True),
    ("end", scene_end, 8.0, False),
]


def timeline():
    t, out = 0.0, []
    for name, fn, dur, wm in SCENES:
        out.append((name, fn, t, dur, wm))
        t += dur - XFADE
    total = t + XFADE
    return out, total


def render_scene(fn, wm, local_t, where):
    c = Canvas()
    fn(c, local_t)
    if wm:
        watermark(c, local_t)
    c.check(where)
    return c.frame()


def frame_at(sec: float) -> Image.Image:
    tl, total = timeline()
    active = [(n, fn, s, d, wm) for n, fn, s, d, wm in tl if s <= sec < s + d]
    if not active:
        active = [tl[-1]]
    imgs = []
    for n, fn, s, d, wm in active:
        imgs.append((render_scene(fn, wm, sec - s, f"{n}@{sec - s:.2f}s"), s, d))
    if len(imgs) == 1:
        return imgs[0][0]
    (a_img, _, _), (b_img, s1, _) = imgs[0], imgs[1]
    k = clamp((sec - s1) / XFADE)
    return Image.blend(a_img, b_img, k)


def _worker(i: int) -> bytes:
    return frame_at(i / FPS).tobytes()


def check_all(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    tl, _ = timeline()
    for n, fn, _s, d, wm in tl:
        # draw at several moments, always the fully-revealed end state last
        for lt in (d * 0.35, d * 0.7, d - 0.05):
            img = render_scene(fn, wm, lt, f"{n}@{lt:.2f}s")
            if lt == d - 0.05:
                img.save(outdir / f"{n}.png")
        print(f"ok {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--still", type=float)
    ap.add_argument("--out", default=str(ROOT / "docs" / "launch.mp4"))
    ap.add_argument("--preview-dir", default=str(ROOT / "assets" / "_preview"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.check:
        check_all(Path(a.preview_dir))
        return
    if a.still is not None:
        p = Path(a.preview_dir)
        p.mkdir(parents=True, exist_ok=True)
        frame_at(a.still).save(p / f"still_{a.still:.1f}.png")
        return
    tl, total = timeline()
    n = int(total * FPS)
    print(f"{n} frames, {total:.1f}s")
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    with Pool(a.workers) as pool:
        for i, buf in enumerate(pool.imap(_worker, range(n), chunksize=6)):
            ff.stdin.write(buf)
            if i % 150 == 0:
                print(f"  frame {i}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
