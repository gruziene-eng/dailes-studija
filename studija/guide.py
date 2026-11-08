"""Tapybos plano generavimas: drobė, paletė, maišymo receptai, tonų studijos, technikos, žingsniai, pirkiniai."""
from __future__ import annotations

import itertools
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .imaging import linear_to_srgb, load_image, rgb_to_hsv, rgb_to_lab, srgb_to_linear

# ------------------------------------------------------------------ dažai
# (pavadinimas LT, pavadinimas EN, sRGB atspalvis, dažymo stiprumas)
PIGMENTS = [
    ("Titano balta", "Titanium White", (247, 247, 243), 1.0),
    ("Kadmio geltona šviesi", "Cadmium Yellow Light (Hue)", (255, 212, 0), 1.1),
    ("Geltonoji ochra", "Yellow Ochre", (200, 145, 48), 1.0),
    ("Kadmio raudona", "Cadmium Red Medium (Hue)", (212, 32, 36), 1.2),
    ("Alizarino raudona", "Alizarin Crimson (Permanent)", (142, 22, 48), 1.4),
    ("Ultramarinas", "Ultramarine Blue", (36, 42, 145), 1.4),
    ("Ftalo mėlyna", "Phthalo Blue", (12, 58, 128), 2.0),
    ("Deginta siena", "Burnt Sienna", (138, 62, 28), 1.0),
    ("Deginta umbra", "Burnt Umber", (78, 52, 36), 1.2),
    ("Peino pilka", "Payne's Grey", (48, 56, 68), 1.8),
]
PIG_LIN = np.clip(srgb_to_linear(np.array([p[2] for p in PIGMENTS], dtype=np.float64)), 0.004, 1)
PIG_STR = np.array([p[3] for p in PIGMENTS])
PARTS = np.array([0, 1, 2, 3, 4, 6, 9, 14])


@lru_cache(maxsize=1)
def _mix_table():
    """Iš anksto apskaičiuoti visi 1–3 dažų mišiniai (svertinis geometrinis vidurkis ≈ subtraktyvus maišymas)."""
    combos, labs = [], []
    logp = np.log(PIG_LIN)
    grid = np.array(list(itertools.product(PARTS, repeat=3)), dtype=np.float64)
    grid = grid[grid.sum(1) > 0]
    for tri in itertools.combinations(range(len(PIGMENTS)), 3):
        idx = np.array(tri)
        w = grid * PIG_STR[idx]
        w = w / w.sum(1, keepdims=True)
        lin = np.exp(w @ logp[idx])
        rgb = linear_to_srgb(lin)
        labs.append(rgb_to_lab(rgb))
        for g in grid:
            combos.append((tri, tuple(int(x) for x in g)))
    labs = np.concatenate(labs)
    n_used = np.array([sum(1 for x in c[1] if x) for c in combos])
    # ar mišinyje yra geltonų (kad mėlynoms spalvoms nesiūlytume geltonos – realiai gautųsi žalia)
    has_yellow = np.array([any(p and t in (1, 2) for t, p in zip(c[0], c[1])) for c in combos])
    return combos, labs, n_used, has_yellow


def mix_recipe(rgb) -> dict:
    combos, labs, n_used, has_yellow = _mix_table()
    target = rgb_to_lab(np.array(rgb, dtype=np.float64))
    de = np.sqrt(((labs - target) ** 2).sum(1))
    cost = de + 2.2 * (n_used - 1)
    h, s_, _ = rgb_to_hsv(np.array(rgb, dtype=np.float64)[None])[0]
    if 185 < h < 300 and s_ > 0.15:
        cost = cost + 25 * has_yellow
    i = int(cost.argmin())
    tri, parts = combos[i]
    items = [(PIGMENTS[t], p, t) for t, p in zip(tri, parts) if p]
    g = math.gcd(*[p for _, p, _ in items]) if items else 1
    total = sum(p for _, p, _ in items)
    items.sort(key=lambda x: -x[1])
    out = []
    for pig, p, t in items:
        share = p / total
        if share < 0.09 and len(items) > 1:
            kiekis = "truputis"
        else:
            kiekis = f"{p // g} d."
        out.append({"dazai": pig[0], "en": pig[1], "dalys": p // g, "kiekis": kiekis, "idx": t,
                    "hex": "#%02x%02x%02x" % pig[2]})
    return {"ingredientai": out, "tikslumas": round(float(de[i]), 1)}


# ------------------------------------------------------------------ paletė
def kmeans_lab(lab: np.ndarray, k: int, iters: int = 12, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = [lab[rng.integers(len(lab))]]
    for _ in range(1, k):  # k-means++
        d = np.min([((lab - c) ** 2).sum(1) for c in centers], axis=0)
        centers.append(lab[rng.choice(len(lab), p=d / d.sum())])
    C = np.array(centers)
    for _ in range(iters):
        lbl = ((lab[:, None, :] - C[None]) ** 2).sum(2).argmin(1)
        for j in range(k):
            m = lbl == j
            if m.any():
                C[j] = lab[m].mean(0)
    return C, lbl


def color_name(rgb) -> str:
    h, s, v = rgb_to_hsv(np.array(rgb, dtype=np.float64)[None])[0]
    if v < 0.16:
        return "beveik juoda"
    if s < 0.12:
        return "balta" if v > 0.88 else ("šviesiai pilka" if v > 0.6 else ("pilka" if v > 0.35 else "tamsiai pilka"))
    names = [(15, "raudona"), (40, "oranžinė"), (65, "geltona"), (160, "žalia"), (195, "žalsvai mėlyna"),
             (255, "mėlyna"), (290, "violetinė"), (335, "rožinė"), (361, "raudona")]
    base = next(n for lim, n in names if h < lim)
    if base in ("oranžinė", "raudona") and v < 0.6 and s < 0.75:
        base = "ruda"
    if base == "geltona" and s < 0.45:
        base = "smėlio"
    pref = "šviesiai " if v > 0.8 and s < 0.45 else ("tamsiai " if v < 0.4 else "")
    if s < 0.3 and pref == "":
        pref = "prislopinta "
    return pref + base


def extract_palette(img: Image.Image, k: int = 9) -> list[dict]:
    small = img.copy()
    small.thumbnail((160, 160))
    arr = np.asarray(small, dtype=np.float64).reshape(-1, 3)
    lab = rgb_to_lab(arr)
    C, lbl = kmeans_lab(lab, k)
    cols = []
    for j in range(k):
        m = lbl == j
        if m.sum() == 0:
            continue
        rgb = arr[m].mean(0)
        cols.append({"lab": C[j], "rgb": rgb, "pct": float(m.mean())})
    # sujungiam labai panašias
    merged = []
    for c in sorted(cols, key=lambda c: -c["pct"]):
        for m in merged:
            if np.sqrt(((m["lab"] - c["lab"]) ** 2).sum()) < 9:
                tot = m["pct"] + c["pct"]
                m["rgb"] = (m["rgb"] * m["pct"] + c["rgb"] * c["pct"]) / tot
                m["lab"] = (m["lab"] * m["pct"] + c["lab"] * c["pct"]) / tot
                m["pct"] = tot
                break
        else:
            merged.append(dict(c))
    merged.sort(key=lambda c: -c["lab"][0])
    out = []
    for i, c in enumerate(merged, 1):
        rgb = tuple(int(round(x)) for x in c["rgb"])
        L = float(c["lab"][0])
        out.append(
            {
                "nr": i,
                "hex": "#%02x%02x%02x" % rgb,
                "rgb": rgb,
                "dalis": round(c["pct"] * 100, 1),
                "tonas": int(min(5, max(1, round(L / 25) + 1))),
                "pavadinimas": color_name(rgb),
                "receptas": mix_recipe(rgb),
            }
        )
    return out


# ------------------------------------------------------------------ drobė
CANVASES = [(20, 20), (20, 30), (24, 30), (30, 30), (30, 40), (30, 60), (35, 50), (40, 40), (40, 50),
            (40, 60), (40, 80), (50, 50), (50, 60), (50, 70), (50, 100), (60, 80), (60, 90), (70, 100),
            (80, 100), (30, 90)]


def canvas_options(w: int, h: int) -> list[dict]:
    landscape = w >= h
    ratio = max(w, h) / min(w, h)
    bands = [("Mažas – greitas eskizas (1–2 vakarai)", 0, 40),
             ("Vidutinis – rekomenduojamas", 41, 70),
             ("Didelis – įspūdingas akcentas", 71, 200)]
    res = []
    for label, lo, hi in bands:
        cands = [c for c in CANVASES if lo <= max(c) <= hi]
        best = min(cands, key=lambda c: abs(max(c) / min(c) - ratio))
        long_, short = max(best), min(best)
        cw, ch = (long_, short) if landscape else (short, long_)
        r = long_ / short
        if abs(r - ratio) < 0.03:
            crop = "Nuotrauka tinka be apkarpymo."
        elif r < ratio:
            lost = 1 - r / ratio
            crop = f"Apkarpykite ~{round(lost*100)}% {'iš šonų' if landscape else 'iš viršaus/apačios'}."
        else:
            lost = 1 - ratio / r
            crop = f"Apkarpykite ~{round(lost*100)}% {'iš viršaus/apačios' if landscape else 'iš šonų'}."
        res.append({"etiketė": label, "plotis": cw, "aukstis": ch, "apkarpymas": crop})
    return res


def crop_to(img: Image.Image, cw: float, ch: float) -> Image.Image:
    W, H = img.size
    target = cw / ch
    if W / H > target:
        nw = int(H * target)
        x = (W - nw) // 2
        return img.crop((x, 0, x + nw, H))
    nh = int(W / target)
    y = (H - nh) // 2
    return img.crop((0, y, W, y + nh))


def grid_cell_cm(cw: int, ch: int) -> int:
    long_ = max(cw, ch)
    return 5 if long_ <= 40 else (10 if long_ <= 80 else 10)


# ------------------------------------------------------------------ studijų vaizdai
def _font(size: int):
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render(kind: str, path: str, cw: int, ch: int, palette: list[dict] | None = None) -> Image.Image:
    img = crop_to(load_image(path, max_side=1400), cw, ch)
    img.thumbnail((1100, 1100), Image.LANCZOS)
    W, H = img.size

    if kind == "tinklelis":
        out = img.copy()
        d = ImageDraw.Draw(out, "RGBA")
        cell = grid_cell_cm(cw, ch)
        nx, ny = cw / cell, ch / cell
        f = _font(max(12, W // 55))
        for i in range(1, math.ceil(nx)):
            x = W * i / nx
            d.line([(x, 0), (x, H)], fill=(255, 255, 255, 200), width=2)
            d.line([(x + 1, 0), (x + 1, H)], fill=(0, 0, 0, 90), width=1)
        for j in range(1, math.ceil(ny)):
            y = H * j / ny
            d.line([(0, y), (W, y)], fill=(255, 255, 255, 200), width=2)
            d.line([(0, y + 1), (W, y + 1)], fill=(0, 0, 0, 90), width=1)
        for i in range(math.ceil(nx)):
            for j in range(math.ceil(ny)):
                lbl = f"{chr(65 + i)}{j + 1}"
                d.text((W * i / nx + 4, H * j / ny + 3), lbl, fill=(255, 255, 255, 230), font=f,
                       stroke_width=2, stroke_fill=(0, 0, 0, 160))
        return out

    if kind == "kompozicija":
        out = img.copy()
        d = ImageDraw.Draw(out, "RGBA")
        for t in (1 / 3, 2 / 3):
            d.line([(W * t, 0), (W * t, H)], fill=(255, 210, 80, 220), width=3)
            d.line([(0, H * t), (W, H * t)], fill=(255, 210, 80, 220), width=3)
        for tx in (1 / 3, 2 / 3):
            for ty in (1 / 3, 2 / 3):
                r = max(6, W // 90)
                d.ellipse([W * tx - r, H * ty - r, W * tx + r, H * ty + r], outline=(255, 90, 60, 255), width=3)
        return out

    if kind in ("tonai3", "tonai5"):
        n = 3 if kind == "tonai3" else 5
        g = img.convert("L").filter(ImageFilter.GaussianBlur(radius=max(2, W // 250)))
        g = g.filter(ImageFilter.MedianFilter(7))
        a = np.asarray(g, dtype=np.float64)
        qs = np.percentile(a, np.linspace(0, 100, n + 1)[1:-1])
        lvl = np.digitize(a, qs)
        tones = np.linspace(40, 235, n)
        return Image.fromarray(tones[lvl].astype(np.uint8)).convert("RGB")

    if kind in ("zonos", "konturai"):
        pal = palette or extract_palette(img)
        # žymim mažesnėje raiškoje – zonos stambesnės, lengviau tapyti
        small = img.copy()
        small.thumbnail((260, 260), Image.LANCZOS)
        base = small.filter(ImageFilter.MedianFilter(5))
        arr = np.asarray(base, dtype=np.float64)
        lab = rgb_to_lab(arr.reshape(-1, 3))
        P = rgb_to_lab(np.array([c["rgb"] for c in pal], dtype=np.float64))
        lbl = ((lab[:, None, :] - P[None]) ** 2).sum(2).argmin(1).reshape(arr.shape[:2])
        lim = Image.fromarray(lbl.astype(np.uint8)).filter(ImageFilter.ModeFilter(5)).filter(ImageFilter.ModeFilter(5))
        lim = lim.resize((W, H), Image.NEAREST).filter(ImageFilter.ModeFilter(9))
        lbl = np.asarray(lim)
        if kind == "zonos":
            cols = np.array([c["rgb"] for c in pal], dtype=np.uint8)
            return Image.fromarray(cols[lbl])
        edge = np.zeros_like(lbl, dtype=bool)
        edge[:, 1:] |= lbl[:, 1:] != lbl[:, :-1]
        edge[1:, :] |= lbl[1:, :] != lbl[:-1, :]
        e = Image.fromarray((~edge * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3))
        return e.convert("RGB")

    if kind == "originalas":
        return img
    raise ValueError(kind)


# ------------------------------------------------------------------ technikos
def techniques(scene: dict, metrics: dict, medium: str) -> list[dict]:
    ak = medium == "akrilas"
    T = []

    def add(key, name, kada, kaip, teptukas):
        T.append({"id": key, "pavadinimas": name, "kada": kada, "kaip": kaip, "teptukas": teptukas})

    add("blokavimas", "Didelių formų blokavimas", "Visada – pirmasis spalvų sluoksnis.",
        ["Prisimerkite ir matykite tik 5–7 dideles dėmes.",
         "Plokščiu plačiu teptuku užpildykite kiekvieną dėmę jos vidutine spalva – be detalių.",
         "Dirbkite nuo tolimiausių planų link artimiausių."],
        "Plokščias Nr. 12–20 arba 2–3 cm pločio")
    if scene["dangus"] > 0.25 or scene["saulelydis"] > 0.08:
        add("gradientas", "Dangaus gradientas „šlapias ant šlapio“",
            "Nuotraukoje daug dangaus" + (" ir saulėlydžio spalvų." if scene["saulelydis"] > 0.08 else "."),
            ["Užtepkite horizonto spalvą apačioje, zenito – viršuje, kol dar šlapia.",
             "Per vidurį sujunkite ilgais horizontaliais X formos potėpiais švariu, vos drėgnu teptuku.",
             ("Akrilas džiūsta greitai: įmaišykite lėtinimo medžiagos (retarder) ir purkškite vandens."
              if ak else "Aliejus leidžia derinti ilgai – švelniai sumaišykite sausu vėduokliniu teptuku."),
             "Dangus prie horizonto visada šviesesnis ir šiltesnis nei viršuje."],
            "Platus plokščias 3–5 cm + švarus vėduoklinis")
        add("debesys", "Debesys – lesiravimas (scumbling)", "Minkštiems debesų kraštams.",
            ["Ant išdžiūvusio dangaus apvaliu sausu teptuku sukamaisiais judesiais užtrinkite baltą su lašu ochros.",
             "Debesų apačią darykite pilkšvai violetinę (balta + ultramarinas + truputis siena).",
             "Kraštus suminkštinkite pirštu ar sausu teptuku – tik saulės pusėje kraštas aštresnis."],
            "Apvalus šeriuotas Nr. 8–10, „filbert“ Nr. 8")
    if scene["vanduo"] > 0.15:
        add("vanduo", "Vanduo ir atspindžiai", "Apatinėje dalyje daug vandens.",
            ["Vandenį tapykite tik horizontaliais potėpiais.",
             "Atspindžiai – vertikalūs, tos pačios formos kaip daiktas, bet šiek tiek tamsesni ir blankesni.",
             "Užbaigus, vienu horizontaliu brūkšniu švariu teptuku lengvai „perbraukite“ atspindžius.",
             "Paskutinės – smulkios baltos blizgės ploniausiu teptuku ar mentele."],
            "Plokščias Nr. 10, „liner“ Nr. 0–1")
    if scene["augmenija"] > 0.12:
        add("lapija", "Lapija – taškavimas (stippling)", "Daug medžių, krūmų ar žolės.",
            ["Pradėkite tamsiausia žalia (ftalo mėlyna + siena + geltona) – tai šešėlinės lapijos masės.",
             "Vėduokliniu ar senu šeriuotu teptuku vertikaliai „baksnokite“ vis šviesesnėmis žaliomis.",
             "Šviesiausius lapus dėkite tik iš saulės pusės – ne visur!",
             "Tolimi medžiai – melsvesni ir blankesni (oro perspektyva)."],
            "Vėduoklinis Nr. 4–6, kempinėlė, seni šeriai")
    if metrics.get("detaliu_tankis", 0) > 0.13:
        add("supaprastinimas", "Supaprastinimas", "Nuotraukoje labai daug smulkmenų.",
            ["Naudokite 3 tonų studiją – ji rodo, ką iš tikrųjų reikia tapyti.",
             "Smulkias detales (langai, akmenys, lapai) parodykite tik 2–3 vietose, ten kur žiūrovo akis.",
             "Likusią dalį palikite apibendrintą – tai atrodo profesionaliau, ne mažiau."],
            "—")
    if scene["saulelydis"] > 0.08 or metrics.get("kontrastas", 0) > 60:
        add("glazura", "Glazūravimas (permatomi sluoksniai)", "Švytinčioms saulėlydžio ar šviesos vietoms.",
            [("Atskieskite dažus su akrilo glazūravimo terpe (1:5) – turi būti permatomi."
              if ak else "Atskieskite dažus su lininiu aliejumi / Liquin – permatomas sluoksnis."),
             "Ant visiškai išdžiūvusio sluoksnio plonai užtepkite šiltą glazūrą (alizarinas, ochra).",
             "Kartokite 2–3 kartus – spalva taps gili ir švytinti."],
            "Minkštas sintetinis plokščias Nr. 10–14")
    add("mentele", "Faktūra mentele (impasto)", "Uolos, bangų puta, žievė, sniegas, ryškūs blyksniai.",
        ["Paimkite tirštų dažų ant mentelės krašto.",
         "Užtepkite trumpu judesiu ir nebeliesite – faktūra turi likti.",
         "Naudokite tik akcentams priekiniame plane."],
        "Tapybos mentelė (lašo formos), Nr. 20–45")
    add("akcentai", "Šviesūs akcentai ir detalės", "Paskutinis etapas.",
        ["Šviesiausi taškai ir ploniausios linijos – pačioje pabaigoje.",
         "Tamsiausi tamsumai – irgi pabaigoje (akrilu) arba pradžioje plonai (aliejumi).",
         "Atsitraukite 3 m – jei detalė nesimato iš toli, ji nereikalinga."],
        "Apvalus Nr. 2, „liner“ Nr. 0")
    return T


def difficulty(metrics: dict, palette: list[dict], photo: dict) -> dict:
    score = 0
    score += min(3, metrics.get("detaliu_tankis", 0.1) / 0.06)
    score += max(0, len(palette) - 5) * 0.4
    if photo.get("faces", 0) or photo.get("persons", 0):
        score += 1.5
    if score < 2.5:
        return {"lygis": 1, "tekstas": "Pradedančiajai", "valandos": "4–8 val."}
    if score < 4:
        return {"lygis": 2, "tekstas": "Vidutinis", "valandos": "8–15 val."}
    return {"lygis": 3, "tekstas": "Pažengusiai", "valandos": "15–30 val."}


# ------------------------------------------------------------------ žingsniai
def steps(metrics, scene, palette, canvas, medium) -> list[dict]:
    ak = medium == "akrilas"
    cw, ch = canvas["plotis"], canvas["aukstis"]
    cell = grid_cell_cm(cw, ch)
    nx, ny = math.ceil(cw / cell), math.ceil(ch / cell)
    hz = metrics.get("horizontas")
    darkest = min(palette, key=lambda c: c["tonas"]) if palette else None
    S = []

    def add(t, txt, items, vaizdas=None, laikas=None):
        S.append({"pavadinimas": t, "aprasas": txt, "punktai": items, "vaizdas": vaizdas, "laikas": laikas})

    add("Pasiruošimas", "Vieta, šviesa, drobė.",
        ["Dirbkite prie dienos šviesos arba po neutralia (5000–6500 K) lempa.",
         "Jei drobė negruntuota – užtepkite 2 sluoksnius gesso, kiekvieną leiskite išdžiūti.",
         ("Tonuokite drobę: deginta siena + daug vandens, plonai, nuvalykite servetėle. Džiūsta ~15 min."
          if ak else "Tonuokite drobę: deginta siena + bekvapis skiediklis, plonai. Džiūsta ~1 val."),
         "Atsispausdinkite arba atsidarykite planšetėje „Tinklelio“ ir „3 tonų“ vaizdus."],
        None, "30 min")
    add("Kompozicijos perkėlimas tinkleliu",
        f"Drobėje pieštuku ar akvareliniu pieštuku nubrėžkite {nx}×{ny} langelių tinklelį ({cell}×{cell} cm).",
        [f"Pažymėkite stulpelius A–{chr(64 + nx)} ir eilutes 1–{ny} kaip vaizde „Tinklelis“.",
         "Perpieškite pagrindines linijas langelis po langelio – žiūrėkite, kur linija kerta langelio kraštą.",
         (f"Horizontas yra maždaug {round(hz*100)}% aukščio nuo viršaus – {round(hz*ch)} cm." if hz else
          "Pirmiausia nubrėžkite ilgiausias, svarbiausias linijas."),
         "Pieškite tik kontūrus (žr. „Kontūrai“), jokių šešėlių. Tinklelį vėliau uždengs dažai."],
        "tinklelis", "45–90 min")
    add("Tonų eskizas (underpainting)", "Viena spalva nustatykite tamsius ir šviesius plotus.",
        ["Atskieskite degintą umbrą (arba sieną + ultramariną) ir užtepkite tamsiausius plotus pagal „3 tonų“ studiją.",
         "Vidutinius tonus palikite tonuotos drobės spalvos, šviesiausius – nuvalykite servetėle.",
         "Šis etapas nusprendžia 70 % paveikslo sėkmės – tikrinkite su studija.",
         ("Akrilas išdžius per 20–30 min." if ak else "Tapykite plonai („liesai“) – vėliau sluoksniai bus riebesni.")],
        "tonai3", "30–60 min")
    order = []
    if scene["dangus"] > 0.2 or scene["saulelydis"] > 0.08:
        order.append("dangus")
    order.append("tolimiausi planai (kalnai, horizontas – blankesni, melsvesni)")
    if scene["vanduo"] > 0.15:
        order.append("vanduo")
    order.append("vidurinis planas")
    order.append("priekinis planas")
    add("Didelių spalvų dėmių blokavimas", "Paletės spalvas tapykite plačiu teptuku nuo galo į priekį.",
        [f"Tvarka: {' → '.join(order)}.",
         "Spalvas sumaišykite iš anksto pagal receptus (skirtukas „Paletė“) – po krūvelę kiekvienai.",
         "Žiūrėkite į „Spalvų zonų“ vaizdą – tai yra būtent tos dėmės.",
         "Kraštus tarp zonų kol kas palikite šiurkščius."],
        "zonos", "1–3 val.")
    add("Tonai, šviesa ir šešėliai", "Antras sluoksnis: perėjimai tarp zonų, šviesos kryptis.",
        ["Nustatykite, iš kur šviečia saulė – šviesioje pusėje spalvos šiltesnės, šešėlyje – vėsesnės.",
         "Sujunkite perėjimus (dangus, vanduo) arba palikite aštrius kraštus ten, kur kontrastas didelis.",
         f"Tamsiausia vieta: {darkest['pavadinimas'] if darkest else 'tamsiausios zonos'} – nenaudokite grynos juodos.",
         "Kas 20 min. atsitraukite ir pažiūrėkite iš toli arba per telefono kamerą nespalvotame režime."],
        "tonai5", "2–4 val.")
    add("Detalės ir akcentai", "Tik dabar – smulkmenos.",
        ["Detalizuokite tik židinio vietą (dažniausiai ties trečdalių susikirtimu – žr. „Kompozicija“).",
         "Šviesiausi blyksniai, ploniausios šakos, bangos putos – pačioje pabaigoje.",
         "Pasirašykite plonu teptuku apatiniame kampe."],
        "kompozicija", "1–3 val.")
    add("Lakavimas", "Apsaugo ir suvienodina blizgesį.",
        [("Akrilas: lakuokite po 2–3 dienų matiniu arba satininiu laku." if ak
          else "Aliejus: lakuokite tik po 6–12 mėnesių; iki tol – tik „retušavimo“ laku."),
         "Lakuokite plačiu minkštu teptuku vienu kryptimi, be dulkių."],
        None, "20 min")
    return S


# ------------------------------------------------------------------ pirkiniai
BASE_SUPPLIES = {
    "akrilas": [
        ("Drobė", "Ištempta gruntuota drobė", "{canvas}", "8–35 €"),
        ("Teptukai", "Plokšti sintetiniai Nr. 4, 8, 12 ir platus 3 cm", "Dideliems plotams ir blokavimui", "10–20 €"),
        ("Teptukai", "„Filbert“ Nr. 6 ir 10", "Švelnūs kraštai, debesys", "6–12 €"),
        ("Teptukai", "Apvalus Nr. 2 ir „liner“ Nr. 0", "Detalėms ir linijoms", "4–8 €"),
        ("Įrankiai", "Tapybos mentelė (lašo formos)", "Maišymui ir faktūrai", "3–8 €"),
        ("Įrankiai", "Drėgnoji paletė („stay-wet“) arba balta plastikinė paletė", "Akrilas palaikomas drėgnas valandas", "8–20 €"),
        ("Įrankiai", "Purškiklis su vandeniu, 2 indeliai vandeniui, popieriniai rankšluosčiai", "Dažai neišdžius ant paletės", "5 €"),
        ("Terpės", "Akrilo lėtinimo terpė (retarder)", "Ilgesnis darbo laikas šlapiu būdu", "6–10 €"),
        ("Terpės", "Glazūravimo terpė (glazing medium)", "Permatomi sluoksniai", "7–12 €"),
        ("Perkėlimas", "Akvarelinis pieštukas / anglis, liniuotė, minkštas trintukas", "Tinkleliui ir kontūrams", "5 €"),
        ("Pabaigai", "Akrilo lakas (matinis arba satininis)", "Apsauga", "8–15 €"),
        ("Pasirinktinai", "Stalinis arba stovintis molbertas", "Patogiau ir mažiau atspindžių", "20–80 €"),
    ],
    "aliejus": [
        ("Drobė", "Ištempta gruntuota drobė (universalus gruntas)", "{canvas}", "8–35 €"),
        ("Teptukai", "Šeriuoti plokšti Nr. 4, 8, 12", "Tiršti dažai, blokavimas", "12–25 €"),
        ("Teptukai", "„Filbert“ Nr. 6 ir 10 (šeriai arba sintetika)", "Švelnūs kraštai", "8–14 €"),
        ("Teptukai", "Vėduoklinis Nr. 4, apvalus Nr. 2, „liner“ Nr. 0", "Derinimas ir detalės", "8–14 €"),
        ("Įrankiai", "Tapybos mentelė, medinė arba stiklinė paletė", "Maišymui ir faktūrai", "8–20 €"),
        ("Terpės", "Bekvapis mineralinis skiediklis (pvz., Gamsol)", "Teptukų valymui ir pirmam sluoksniui", "8–14 €"),
        ("Terpės", "Lininis aliejus arba Liquin", "Viršutiniams „riebiems“ sluoksniams ir glazūroms", "6–12 €"),
        ("Įrankiai", "Sandarus indelis skiedikliui, skudurai, pirštinės", "Saugumas ir švara", "6 €"),
        ("Perkėlimas", "Anglis arba akvarelinis pieštukas, liniuotė", "Tinkleliui ir kontūrams", "5 €"),
        ("Pabaigai", "Retušavimo lakas, vėliau – galutinis lakas", "Apsauga", "10–18 €"),
        ("Pasirinktinai", "Molbertas", "Patogiau dirbti", "20–80 €"),
    ],
}


def supplies(palettes: list[list[dict]], canvases: list[dict], medium: str) -> dict:
    use = {}
    for pal in palettes:
        for c in pal:
            for ing in c["receptas"]["ingredientai"]:
                d = use.setdefault(ing["idx"], {"kiek": 0.0, "spalvos": 0})
                d["kiek"] += c["dalis"] * ing["dalys"]
                d["spalvos"] += 1
    use.setdefault(0, {"kiek": 1.0, "spalvos": 0})
    largest = max((cv["plotis"] * cv["aukstis"] for cv in canvases), default=1600)
    paints = []
    for idx, u in sorted(use.items(), key=lambda kv: -kv[1]["kiek"]):
        pig = PIGMENTS[idx]
        big = idx == 0 or u["kiek"] > 400
        if idx == 0:
            tube = "200–250 ml" if largest >= 2500 or len(canvases) > 1 else "120 ml"
        else:
            tube = "100–120 ml" if big and largest >= 2500 else ("60–75 ml" if medium == "akrilas" else "37–40 ml")
        paints.append({"dazai": pig[0], "en": pig[1], "hex": "#%02x%02x%02x" % pig[2], "tuba": tube,
                       "kiek_spalvu": u["spalvos"]})
    canv_txt = ", ".join(f"{c['plotis']}×{c['aukstis']} cm" for c in canvases) or "pagal pasirinkimą"
    items = [{"kategorija": k, "daiktas": d, "kam": why.format(canvas=canv_txt), "kaina": p}
             for k, d, why, p in BASE_SUPPLIES[medium]]
    return {"dazai": paints, "kita": items,
            "pastaba": ("Užtenka „studentiškos“ kokybės dažų (pvz., Winsor & Newton Galeria, Amsterdam Standard, "
                        "Liquitex Basics). Kainos – orientacinės." if medium == "akrilas" else
                        "Užtenka „studentiškos“ kokybės dažų (pvz., Winsor & Newton Winton, Van Gogh, Talens). "
                        "Kainos – orientacinės.")}


def build_guide(photo: dict, metrics: dict, medium: str, canvas_idx: int = 1) -> dict:
    img = load_image(photo["path"], max_side=900)
    opts = canvas_options(*img.size)
    canvas = opts[min(canvas_idx, len(opts) - 1)]
    palette = extract_palette(crop_to(img, canvas["plotis"], canvas["aukstis"]))
    return {
        "drobes": opts,
        "drobe": canvas,
        "drobes_idx": canvas_idx,
        "medziaga": medium,
        "paletė": palette,
        "sunkumas": difficulty(metrics, palette, photo),
        "technikos": techniques(metrics["scena"], metrics, medium),
        "zingsniai": steps(metrics, metrics["scena"], palette, canvas, medium),
        "pirkiniai": supplies([palette], [canvas], medium),
        "tinklelis_cm": grid_cell_cm(canvas["plotis"], canvas["aukstis"]),
    }
