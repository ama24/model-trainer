"""Draw labels and predictions on images (boxes, masks, keypoints, rotated boxes) and make thumbnails."""

from __future__ import annotations

import colorsys
import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .formats import Item, Obj


def colour(cls: int) -> tuple[int, int, int]:
    """A distinct, stable colour per class id (golden-ratio hues)."""
    h = (cls * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw(
    image: Image.Image,
    objects: Sequence[Obj],
    names: dict[int, str],
    layers: Iterable[str] = ("boxes", "masks", "keypoints", "labels"),
    scores: Sequence[float] | None = None,
    line: int | None = None,
) -> Image.Image:
    """Overlay objects (normalised coordinates) on an RGB image."""
    layers = set(layers)
    img = image.convert("RGB")
    w, h = img.size
    lw = line or max(1, round(max(w, h) / 400))
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(over)
    d = ImageDraw.Draw(img)
    font = _font(max(10, round(max(w, h) / 60)))
    for i, o in enumerate(objects):
        c = colour(o.cls)
        if "masks" in layers and o.polygon and len(o.polygon) >= 6:
            pts = [
                (o.polygon[j] * w, o.polygon[j + 1] * h) for j in range(0, len(o.polygon) - 1, 2)
            ]
            od.polygon(pts, fill=(*c, 90), outline=(*c, 255))
        if o.obb and "boxes" in layers:
            pts = [(o.obb[j] * w, o.obb[j + 1] * h) for j in range(0, 8, 2)]
            d.line([*pts, pts[0]], fill=c, width=lw)
        elif o.box and "boxes" in layers:
            cx, cy, bw, bh = o.box
            d.rectangle(
                [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
                outline=c,
                width=lw,
            )
        if "keypoints" in layers and o.keypoints:
            r = lw * 2 + 1
            for x, y, v in o.keypoints:
                if v > 0:
                    d.ellipse(
                        [x * w - r, y * h - r, x * w + r, y * h + r], fill=c, outline=(0, 0, 0)
                    )
        if "labels" in layers and o.box:
            text = names.get(o.cls, str(o.cls))
            if scores is not None:
                text += f" {scores[i]:.2f}"
            x0, y0 = (o.box[0] - o.box[2] / 2) * w, (o.box[1] - o.box[3] / 2) * h
            tb = d.textbbox((x0, y0), text, font=font)
            ty = y0 - (tb[3] - tb[1]) - 2 if y0 > (tb[3] - tb[1]) + 2 else y0
            tb = d.textbbox((x0, ty), text, font=font)
            d.rectangle(tb, fill=c)
            d.text((x0, ty), text, fill=(0, 0, 0), font=font)
    img.paste(Image.alpha_composite(img.convert("RGBA"), over).convert("RGB"))
    return img


def render_item(
    it: Item,
    names: dict[int, str],
    layers: Iterable[str] = ("boxes", "masks", "keypoints", "labels"),
    max_side: int | None = None,
) -> Image.Image:
    with Image.open(it.image) as im:
        img = im.convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    out = draw(img, it.objects, names, layers)
    if it.objects and it.objects[0].box is None and names:  # classification: class name banner
        dd = ImageDraw.Draw(out)
        font = _font(max(12, out.size[0] // 14))
        text = names.get(it.objects[0].cls, "?")
        tb = dd.textbbox((4, 4), text, font=font)
        dd.rectangle(tb, fill=colour(it.objects[0].cls))
        dd.text((4, 4), text, fill=(0, 0, 0), font=font)
    return out


def thumbnail(
    it: Item,
    names: dict[int, str],
    cache_dir: Path,
    side: int = 320,
    layers: Sequence[str] = ("boxes", "masks", "keypoints"),
) -> Path:
    """Cached overlay thumbnail (re-made when the image or its labels change)."""
    try:
        mtime = it.image.stat().st_mtime
    except OSError:
        mtime = 0
    sig = f"{it.image}|{mtime}|{side}|{','.join(layers)}|{[(o.cls, o.box) for o in it.objects]}"
    path = Path(cache_dir) / "thumbs" / (hashlib.sha1(sig.encode()).hexdigest() + ".jpg")
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            img = render_item(it, names, layers, max_side=side)
        except OSError:
            img = Image.new("RGB", (side, side), (60, 60, 60))
            ImageDraw.Draw(img).text((10, 10), "missing image", fill=(255, 255, 255))
        img.save(path, quality=85)
    return path
