#!/usr/bin/env python3
"""
Generate public/header.png and public/main.png (GitHub renders PNG reliably).

  pip install -r scripts/requirements-readme-assets.txt
  python3 scripts/gen_readme_png.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"

C_BLUE = "#116fb8"
C_TEAL = "#3be5c3"
C_MUTED = "#444444"
C_CARD_LT = "#d2f3fc"
C_CARD_RT = "#eaf3ff"
C_PANEL = "#fafdff"


def _font(size: int, bold: bool) -> ImageFont.ImageFont:
    paths = (
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        try:
            return ImageFont.truetype(str(p), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def h_gradient(w: int, h: int, left: str, right: str) -> Image.Image:
    lo, hi = _hex(left), _hex(right)
    row = [_lerp(lo, hi, i / max(w - 1, 1)) for i in range(w)]
    buf = []
    for _ in range(h):
        for px in row:
            buf.extend(px)
    return Image.frombytes("RGB", (w, h), bytes(buf))


def paint_gradient_text(
    base: Image.Image,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    bbox = ld.textbbox((0, 0), text, font=font)
    tw = max(bbox[2] - bbox[0], 1)
    th = max(bbox[3] - bbox[1], 1)
    grad = h_gradient(tw, th, C_BLUE, C_TEAL)
    mask = Image.new("L", (tw, th), 0)
    md = ImageDraw.Draw(mask)
    ox, oy = -bbox[0], -bbox[1]
    md.text((ox, oy), text, font=font, fill=255)
    gx0, gy0 = int(x + bbox[0]), int(y + bbox[1])
    region = Image.new("RGB", (tw, th))
    region.paste(grad, mask=mask)
    base.paste(region, (gx0, gy0))


def rounded_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], r: int, **kw) -> None:
    draw.rounded_rectangle(box, radius=r, **kw)


def build_header(scale_w: int = 1800) -> Image.Image:
    w = scale_w
    scale = w / 900
    h = int(438 * scale)
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    title = _font(int(26 * scale), True)
    sub = _font(int(17 * scale), False)
    ctitle = _font(int(19 * scale), True)
    csemi = _font(int(16 * scale), True)
    cbody = _font(int(15 * scale), False)

    cx = w // 2
    y1 = int(38 * scale)
    lt1 = "T2PO: Uncertainty-Guided Exploration Control"
    b1 = d.textbbox((0, 0), lt1, font=title)
    paint_gradient_text(img, (cx - (b1[2] - b1[0]) // 2, y1), lt1, title)

    lt2 = "for Stable Multi-Turn Agentic RL"
    b2 = d.textbbox((0, 0), lt2, font=title)
    paint_gradient_text(img, (cx - (b2[2] - b2[0]) // 2, y1 + int(39 * scale)), lt2, title)

    ys = y1 + int(88 * scale)
    star = "\u2605 "
    tail = "ICML 2026 Spotlight Paper   |   Now Open Source!"
    bx = d.textbbox((0, 0), star + tail, font=sub)
    x0 = cx - (bx[2] - bx[0]) // 2
    d.text((x0, ys), star, font=sub, fill=_hex(C_BLUE))

    xa = x0 + (d.textbbox((0, 0), star, font=sub)[2] - d.textbbox((0, 0), star, font=sub)[0])
    part1 = "ICML 2026 Spotlight Paper   "
    part2 = "|   "
    part3 = "Now Open Source!"
    d.text((xa, ys), part1, font=sub, fill=_hex(C_BLUE))
    xb = xa + d.textbbox((0, 0), part1, font=sub)[2] - d.textbbox((0, 0), part1, font=sub)[0]
    d.text((xb, ys), part2, font=sub, fill=_hex("#666666"))
    xc = xb + d.textbbox((0, 0), part2, font=sub)[2] - d.textbbox((0, 0), part2, font=sub)[0]
    d.text((xc, ys), part3, font=sub, fill=_hex(C_TEAL))

    cw, ch = int(720 * scale), int(152 * scale)
    x0c = cx - cw // 2
    yc = int(168 * scale)
    card = h_gradient(cw, ch, C_CARD_LT, C_CARD_RT)
    m = Image.new("L", (cw, ch), 0)
    md = ImageDraw.Draw(m)
    md.rounded_rectangle([0, 0, cw, ch], radius=int(16 * scale), fill=255)
    img.paste(card, (x0c, yc), m)

    spark = "\u2728"
    sf = _font(int(32 * scale), False)
    d.text((int(128 * scale), yc + int(58 * scale)), spark, font=sf, fill=_hex(C_BLUE))

    tx = int(180 * scale)
    ty = yc + int(28 * scale)
    d.text((tx, ty), "Core Insights: Hesitation is defeat!", font=ctitle, fill=_hex(C_BLUE))
    d.text((tx, ty + int(30 * scale)), "Exploration must be encouraged at every step.", font=csemi, fill=_hex(C_BLUE))
    d.text(
        (tx, ty + int(56 * scale)),
        "Without active exploration, agents become unstable and stagnate.",
        font=cbody,
        fill=_hex(C_MUTED),
    )
    return img


def draw_arrow(draw: ImageDraw.ImageDraw, x: int, y: int, length: int, sw: float) -> None:
    draw.line([(x, y), (x + length, y)], fill=_hex(C_BLUE), width=max(2, int(2 * sw)))
    ah = max(7, int(8 * sw))
    draw.polygon([(x + length, y), (x + length - ah, y - ah // 2), (x + length - ah, y + ah // 2)], fill=_hex(C_BLUE))


def build_main(scale_w: int = 1660) -> Image.Image:
    w = scale_w
    sx = w / 830.0
    h = int(460 * sx)
    img = Image.new("RGB", (w, h), _hex(C_PANEL))
    d = ImageDraw.Draw(img)

    bar_h = max(4, round(8 * sx))
    bar = h_gradient(w, bar_h, C_BLUE, C_TEAL)
    img.paste(bar, (0, 0))

    t_big = _font(round(21 * sx), True)
    t_sub = _font(round(13 * sx), False)
    bx = _font(round(11 * sx), True)
    sm = _font(round(10 * sx), False)

    cx = w // 2
    h1 = "T2PO — Uncertainty-guided exploration control"
    b = d.textbbox((0, 0), h1, font=t_big)
    d.text((cx - (b[2] - b[0]) // 2, round(26 * sx)), h1, font=t_big, fill=_hex("#145a96"))
    h2 = "Token- and turn-level signals stabilize multi-turn agentic RL"
    b2 = d.textbbox((0, 0), h2, font=t_sub)
    d.text((cx - (b2[2] - b2[0]) // 2, round(52 * sx)), h2, font=t_sub, fill=_hex("#5a6878"))

    yb = round(115 * sx)
    bh = round(88 * sx)
    mid_y = yb + bh // 2
    rr = round(11 * sx)

    def box(x_px: float, bw_px: float, lines: list[tuple[str, bool]], teal: bool = False) -> None:
        x_i, bw_i = round(x_px), round(bw_px)
        oc = _hex(C_TEAL) if teal else _hex("#c8e8f2")
        rounded_rect(d, [x_i, yb, x_i + bw_i, yb + bh], rr, fill="white", outline=oc, width=2 if teal else 1)
        yy = yb + round(14 * sx)
        for txt, bol in lines:
            fnt = bx if bol else sm
            col = _hex(C_BLUE) if bol else _hex("#444444")
            d.text((x_i + bw_i // 2, yy), txt, font=fnt, fill=col, anchor="mt")
            yy += round(21 * sx) if bol else round(16 * sx)

    # Layout matches public/main.svg (830 design width)
    box(40 * sx, 150 * sx, [("Multi-turn LLM agent", True), ("(interactive env)", False)])
    box(230 * sx, 170 * sx, [("Token uncertainty", True), ("Interventions when marginal u.c. is low", False)])
    box(440 * sx, 170 * sx, [("Turn exploration", True), ("Resample low-progress turns", False)])
    box(650 * sx, 140 * sx, [("Stable multi-turn", True), ("agentic RL", False)], teal=True)

    def gap_arrow(left_edge: float, left_w: float, right_edge: float) -> None:
        x0 = round((left_edge + left_w) * sx + 6 * sx)
        x1 = round(right_edge * sx - 6 * sx)
        ln = max(8, x1 - x0)
        draw_arrow(d, x0, mid_y, ln, sx)

    gap_arrow(40, 150, 230)
    gap_arrow(230, 170, 440)
    gap_arrow(440, 170, 650)

    py = round(265 * sx)
    ph = round(148 * sx)
    pw = round(650 * sx)
    px = cx - pw // 2
    rounded_rect(d, [px, py, px + pw, py + ph], round(13 * sx), fill=_hex("#eef8fc"), outline=_hex("#bde4f2"))
    cap = _font(round(13 * sx), True)
    d.text((cx, py + round(34 * sx)), "Evaluation & benchmarks (representative targets)", font=cap, fill=_hex(C_BLUE), anchor="mt")
    d.text(
        (cx, py + round(62 * sx)),
        "WebShop · ALFWorld · SearchQA · embodied & web/search agents",
        font=sm,
        fill=_hex("#3a4652"),
        anchor="mt",
    )
    d.text(
        (cx, py + round(104 * sx)),
        "PNG from scripts/gen_readme_png.py — replace with your teaser figure anytime",
        font=sm,
        fill=_hex("#6a7580"),
        anchor="mt",
    )

    return img


def main() -> None:
    PUBLIC.mkdir(parents=True, exist_ok=True)
    build_header(1800).save(PUBLIC / "header.png", optimize=True)
    build_main(1660).save(PUBLIC / "main.png", optimize=True)
    print("Wrote", PUBLIC / "header.png")
    print("Wrote", PUBLIC / "main.png")


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("Install Pillow: pip install -r scripts/requirements-readme-assets.txt", file=sys.stderr)
        sys.exit(1)
