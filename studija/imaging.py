"""Vaizdo įkėlimas ir kokybės / tapybiškumo įvertinimas (viskas lokaliai, be interneto)."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

warnings.filterwarnings("ignore")

try:  # HEIC (iPhone) palaikymas
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pass

Image.MAX_IMAGE_PIXELS = 300_000_000


def load_image(path: str | Path, max_side: int | None = None) -> Image.Image:
    img = Image.open(path)
    if max_side and img.format == "JPEG":
        img.draft("RGB", (max_side, max_side))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    if max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


# ---------------------------------------------------------------- spalvų erdvės
def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055) * 255.0


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(np.asarray(rgb, dtype=np.float64))
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    a = arr / 255.0
    mx, mn = a.max(-1), a.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    nz = d > 1e-6
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    h = h * 60
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
    return np.stack([h, s, mx], axis=-1)


# ---------------------------------------------------------------- požymiai
def dhash(img: Image.Image, size: int = 8) -> int:
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    v = 0
    for bit in bits:
        v = (v << 1) | int(bit)
    return v


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def scene_features(arr: np.ndarray) -> dict:
    """Paprasta scenos analizė: dangus, vanduo, augmenija, saulėlydis, sniegas, naktis."""
    hsv = rgb_to_hsv(arr.astype(np.float64))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    H = arr.shape[0]
    top = slice(0, H // 3)
    bottom = slice(H // 2, H)

    blueish = (h > 185) & (h < 255) & (s > 0.12) & (v > 0.35)
    bright_low_sat = (s < 0.12) & (v > 0.75)
    green = (h > 65) & (h < 170) & (s > 0.18) & (v > 0.12)
    warm_sky = ((h < 45) | (h > 320)) & (s > 0.3) & (v > 0.45)

    sky_top = float((blueish | bright_low_sat)[top].mean())
    sunset = float(warm_sky[top].mean())
    water = float(blueish[bottom].mean())
    veg = float(green.mean())
    snow = float(((s < 0.1) & (v > 0.85))[bottom].mean())
    brightness = float(v.mean())
    sat = float(s.mean())
    return {
        "dangus": round(sky_top, 3),
        "saulelydis": round(sunset, 3),
        "vanduo": round(water, 3),
        "augmenija": round(veg, 3),
        "sniegas": round(snow, 3),
        "sviesumas": round(brightness, 3),
        "sodrumas": round(sat, 3),
        "naktis": brightness < 0.25,
    }


def horizon_row(gray: np.ndarray) -> float | None:
    """Ieško stipriausio horizontalaus perėjimo (horizonto) – grąžina santykinę aukštį 0..1."""
    rows = gray.mean(axis=1)
    if len(rows) < 20:
        return None
    k = np.ones(5) / 5
    sm = np.convolve(rows, k, mode="same")
    grad = np.abs(np.diff(sm))
    lo, hi = int(len(grad) * 0.12), int(len(grad) * 0.88)
    seg = grad[lo:hi]
    if seg.size == 0 or seg.max() < 4:
        return None
    return round((lo + int(seg.argmax())) / len(rows), 3)


def analyze(path: str | Path) -> dict:
    """Grąžina kokybės metrikas ir tapybiškumo balą (0–100)."""
    img = load_image(path, max_side=512)
    arr = np.asarray(img, dtype=np.float64)
    gray = arr @ np.array([0.299, 0.587, 0.114])

    # aštrumas: Laplaso dispersija
    lap = (
        -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    sharp = float(lap.var())
    # ekspozicija
    clip_lo = float((gray < 8).mean())
    clip_hi = float((gray > 247).mean())
    mean = float(gray.mean())
    contrast = float(gray.std())
    # spalvingumas (Hasler–Süsstrunk)
    rg = arr[..., 0] - arr[..., 1]
    yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    colorful = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    # detalių tankis (per daug smulkmenų = sunku tapyti)
    edges = np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float64)
    edge_density = float((edges > 40).mean())

    scene = scene_features(arr)

    # odos atspalvio dalis centre (portretai / asmenukės) – YCbCr taisyklė
    Hh, Ww = gray.shape
    c = arr[Hh // 6: Hh * 5 // 6, Ww // 5: Ww * 4 // 5]
    cb = 128 - 0.168736 * c[..., 0] - 0.331264 * c[..., 1] + 0.5 * c[..., 2]
    cr = 128 + 0.5 * c[..., 0] - 0.418688 * c[..., 1] - 0.081312 * c[..., 2]
    skin = float(((cb > 80) & (cb < 125) & (cr > 136) & (cr < 172)).mean())
    # dokumentai / ekrano nuotraukos / kvitai: daug blyškių vienodų plotų + tekstas
    hsv = rgb_to_hsv(arr)
    pale = float(((hsv[..., 1] < 0.22) & (hsv[..., 2] > 0.55)).mean())
    nature_px = scene["dangus"] + scene["augmenija"] + scene["vanduo"]
    is_doc = pale > 0.45 and edge_density > 0.07 and nature_px < 0.35

    s_sharp = min(1.0, np.log1p(sharp) / np.log1p(600))
    s_expo = max(0.0, 1 - abs(mean - 120) / 120) * max(0.0, 1 - 4 * (clip_lo + clip_hi))
    s_contrast = min(1.0, contrast / 60)
    s_color = min(1.0, colorful / 70)
    # tapybai geriausia vidutinis detalumas (~6–14 %)
    s_simplicity = max(0.0, 1 - abs(edge_density - 0.10) / 0.18)
    nature = min(1.0, scene["dangus"] + scene["vanduo"] + scene["augmenija"] + scene["saulelydis"])

    quality = 0.40 * s_sharp + 0.35 * s_expo + 0.25 * s_contrast
    paintability = 0.30 * s_color + 0.25 * s_simplicity + 0.30 * nature + 0.15 * s_contrast
    if is_doc:
        paintability *= 0.3
    if skin > 0.12:  # kelionių peizažams žmonės kadre – minusas
        paintability *= max(0.45, 1 - 1.6 * (skin - 0.12))
    w, h = img.size
    return {
        "w": w,
        "h": h,
        "astrumas": round(sharp, 1),
        "kontrastas": round(contrast, 1),
        "spalvingumas": round(colorful, 1),
        "detaliu_tankis": round(edge_density, 3),
        "ekspozicija": round(s_expo, 3),
        "kokybe": round(quality * 100, 1),
        "tapybiskumas": round(paintability * 100, 1),
        "scena": scene,
        "oda": round(skin, 3),
        "dokumentas": bool(is_doc),
        "horizontas": horizon_row(gray),
        "dhash": dhash(img),
    }


def make_thumb(path: str | Path, out: Path, side: int = 480) -> None:
    img = load_image(path, max_side=side)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "JPEG", quality=85)
