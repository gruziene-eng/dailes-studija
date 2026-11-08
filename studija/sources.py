"""Nuotraukų šaltiniai: macOS Photos (iCloud) biblioteka arba paprastas aplankas."""
from __future__ import annotations

import hashlib
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image

from .imaging import load_image  # noqa: F401  (užregistruoja HEIC)

PHOTOS_LIB = Path.home() / "Pictures" / "Photos Library.photoslibrary"
IMG_EXT = {".jpg", ".jpeg", ".heic", ".heif", ".png", ".tif", ".tiff", ".webp"}


# ------------------------------------------------------------------ Photos
def photos_status() -> dict:
    if not PHOTOS_LIB.exists():
        return {"ok": False, "priezastis": "nerasta", "kelias": str(PHOTOS_LIB)}
    try:
        os.listdir(PHOTOS_LIB / "database")
    except PermissionError:
        return {"ok": False, "priezastis": "leidimas", "kelias": str(PHOTOS_LIB)}
    except FileNotFoundError:
        return {"ok": False, "priezastis": "nerasta", "kelias": str(PHOTOS_LIB)}
    try:
        import osxphotos  # noqa: F401
    except Exception as e:  # pragma: no cover
        return {"ok": False, "priezastis": "osxphotos", "klaida": str(e)}
    return {"ok": True, "kelias": str(PHOTOS_LIB)}


def _best_local_path(p) -> str | None:
    """Originalas, jei parsiųstas; kitaip – didžiausia lokali peržiūros kopija (iCloud optimizuotos bibliotekos atveju)."""
    for attr in ("path_edited", "path"):
        try:
            v = getattr(p, attr)
            if v and os.path.exists(v):
                return v
        except Exception:
            pass
    try:
        ders = [d for d in (p.path_derivatives or []) if os.path.exists(d)]
        if ders:
            return max(ders, key=lambda d: os.path.getsize(d))
    except Exception:
        pass
    return None


def scan_photos(progress, since_year: int | None = None) -> list[dict]:
    import osxphotos

    progress("Atidaroma Photos biblioteka…", 0)
    db = osxphotos.PhotosDB(dbfile=str(PHOTOS_LIB))
    photos = db.photos(images=True, movies=False)
    out = []
    n = len(photos)
    for i, p in enumerate(photos):
        if i % 500 == 0:
            progress(f"Skaitomi metaduomenys: {i}/{n}", 5 + 20 * i / max(n, 1))
        try:
            if getattr(p, "hidden", False) or getattr(p, "intrash", False):
                continue
            if getattr(p, "screenshot", False):
                continue
            d = p.date
            if since_year and d and d.year < since_year:
                continue
            path = _best_local_path(p)
            if not path:
                continue
            lat, lon = (p.location or (None, None))
            place = None
            country = None
            try:
                if p.place:
                    names = p.place.names
                    city = (names.city or names.sub_administrative_area or [None])
                    place = city[0] if city else None
                    country = p.place.country_code
                    if not place:
                        place = p.place.name.split(",")[0] if p.place.name else None
            except Exception:
                pass
            score = None
            try:
                sc = p.score
                score = float(sc.overall)
            except Exception:
                pass
            try:
                albums = [a for a in p.albums][:5]
            except Exception:
                albums = []
            out.append(
                {
                    "id": p.uuid,
                    "path": path,
                    "date": d.replace(tzinfo=None).isoformat() if d else None,
                    "lat": lat,
                    "lon": lon,
                    "place": place,
                    "country": country,
                    "favorite": bool(getattr(p, "favorite", False)),
                    "apple_score": score,
                    "persons": len([x for x in (p.persons or []) if x and x != "_UNKNOWN_"]),
                    "faces": len(getattr(p, "face_info", []) or []),
                    "selfie": bool(getattr(p, "selfie", False)),
                    "albums": albums,
                    "filename": p.original_filename,
                    "folder": None,
                }
            )
        except Exception:
            continue
    progress(f"Rasta {len(out)} nuotraukų", 25)
    return out


# ------------------------------------------------------------------ aplankas
def _gps_to_deg(v, ref):
    try:
        d, m, s = [float(x) for x in v]
        deg = d + m / 60 + s / 3600
        return -deg if ref in ("S", "W") else deg
    except Exception:
        return None


def _exif_info(path: Path):
    date = lat = lon = None
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            dt = ex.get(36867) or ex.get(306)
            try:
                sub = ex.get_ifd(0x8769)
                dt = sub.get(36867) or dt
            except Exception:
                pass
            if dt:
                try:
                    date = datetime.strptime(str(dt)[:19], "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    pass
            try:
                gps = ex.get_ifd(0x8825)
                if gps and 2 in gps and 4 in gps:
                    lat = _gps_to_deg(gps[2], gps.get(1))
                    lon = _gps_to_deg(gps[4], gps.get(3))
            except Exception:
                pass
    except Exception:
        return None, None, None
    if date is None:
        date = datetime.fromtimestamp(path.stat().st_mtime)
    return date, lat, lon


def scan_folder(folder: str, progress, since_year: int | None = None) -> list[dict]:
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise ValueError(f"Aplankas nerastas: {root}")
    progress("Ieškoma nuotraukų aplanke…", 0)
    files = [
        p
        for p in root.rglob("*")
        if p.suffix.lower() in IMG_EXT and not p.name.startswith(".") and ".photoslibrary" not in str(p)
    ]
    out = []
    for i, f in enumerate(files):
        if i % 50 == 0:
            progress(f"Skaitomi EXIF duomenys: {i}/{len(files)}", 25 * i / max(len(files), 1))
        try:
            if f.stat().st_size < 60_000:  # piktogramos, ekrano nuotraukų nuotrupos
                continue
        except OSError:
            continue
        date, lat, lon = _exif_info(f)
        if since_year and date and date.year < since_year:
            continue
        rel = f.parent.relative_to(root)
        out.append(
            {
                "id": hashlib.sha1(str(f).encode()).hexdigest()[:16],
                "path": str(f),
                "date": date.isoformat() if date else None,
                "lat": lat,
                "lon": lon,
                "place": None,
                "country": None,
                "favorite": False,
                "apple_score": None,
                "persons": 0,
                "faces": 0,
                "selfie": False,
                "albums": [],
                "filename": f.name,
                "folder": str(rel) if str(rel) != "." else root.name,
            }
        )
    progress(f"Rasta {len(out)} nuotraukų", 25)
    return out


# ------------------------------------------------------------------ kelionės
def _km(a, b):
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


MONTHS = ["sausio", "vasario", "kovo", "balandžio", "gegužės", "birželio", "liepos",
          "rugpjūčio", "rugsėjo", "spalio", "lapkričio", "gruodžio"]


def _range_label(d1: datetime, d2: datetime) -> str:
    if d1.date() == d2.date():
        return f"{d1.year} m. {MONTHS[d1.month-1]} {d1.day} d."
    if d1.year == d2.year and d1.month == d2.month:
        return f"{d1.year} m. {MONTHS[d1.month-1]} {d1.day}–{d2.day} d."
    return f"{d1:%Y-%m-%d} – {d2:%Y-%m-%d}"


def group_trips(photos: list[dict], mode: str = "auto", min_photos: int = 6,
                home_radius_km: float = 60, gap_days: float = 2.5) -> list[dict]:
    """Grupuoja nuotraukas į keliones.

    auto  – pagal GPS: randama „namų“ vieta (dažniausia), kelionė = nuotraukos >60 km nuo namų,
            sujungiamos į vieną, jei tarpas tarp jų < 2,5 d. Be GPS – pagal laiko tarpus.
    album – kiekvienas Photos albumas = kelionė.
    folder – kiekvienas poaplankis = kelionė.
    """
    dated = [p for p in photos if p.get("date")]
    for p in dated:
        p["_dt"] = datetime.fromisoformat(p["date"])
    groups: dict[str, list[dict]] = defaultdict(list)

    if mode == "album":
        for p in dated:
            for a in p.get("albums") or []:
                groups[a].append(p)
    elif mode == "folder":
        for p in dated:
            groups[p.get("folder") or "Aplankas"].append(p)
    else:
        located = [p for p in dated if p.get("lat") is not None and p.get("lon") is not None
                   and not (p["lat"] == 0 and p["lon"] == 0)]
        home = None
        if len(located) >= 20:
            cells = Counter()
            seen = set()
            for p in located:
                cell = (round(p["lat"] * 2) / 2, round(p["lon"] * 2) / 2)
                key = (cell, p["_dt"].date())
                if key not in seen:  # skaičiuojam dienas, ne kadrus
                    seen.add(key)
                    cells[cell] += 1
            home = cells.most_common(1)[0][0]

        def away(p):
            if home is None:
                return True  # be GPS – viską laikom kandidatais
            if p.get("lat") is None:
                return None  # nežinoma
            return _km((p["lat"], p["lon"]), home) > home_radius_km

        seq = sorted(dated, key=lambda p: p["_dt"])
        cur: list[dict] = []
        last_away_dt = None
        trips = []
        for p in seq:
            a = away(p)
            if a is None:
                # be GPS – priskiriam, jei nuotrauka einamosios kelionės laikotarpyje
                if cur and last_away_dt and p["_dt"] - last_away_dt < timedelta(days=1):
                    cur.append(p)
                continue
            if a:
                if cur and last_away_dt and p["_dt"] - last_away_dt > timedelta(days=gap_days):
                    trips.append(cur)
                    cur = []
                cur.append(p)
                last_away_dt = p["_dt"]
        if cur:
            trips.append(cur)
        if home is None:
            # be GPS: laikom „kelione“ tik tankius kelių dienų pliūpsnius
            trips = [t for t in trips if len(t) >= max(min_photos, 15)
                     and (t[-1]["_dt"] - t[0]["_dt"]).days >= 1]
        for t in trips:
            places = Counter(p["place"] for p in t if p.get("place"))
            folders = Counter(p["folder"] for p in t if p.get("folder"))
            d1, d2 = t[0]["_dt"], t[-1]["_dt"]
            if places:
                top = [n for n, _ in places.most_common(3)]
                name = ", ".join(top)
            elif folders and home is None:
                name = folders.most_common(1)[0][0]
            else:
                name = "Kelionė"
            groups[f"{name}|{d1.isoformat()}"] = t

    trips_out = []
    for key, items in groups.items():
        if len(items) < min_photos:
            continue
        items.sort(key=lambda p: p["_dt"])
        d1, d2 = items[0]["_dt"], items[-1]["_dt"]
        name = key.split("|")[0]
        countries = Counter(p["country"] for p in items if p.get("country"))
        tid = hashlib.sha1(f"{mode}{key}".encode()).hexdigest()[:10]
        trips_out.append(
            {
                "id": tid,
                "pavadinimas": name,
                "datos": _range_label(d1, d2),
                "pradzia": d1.isoformat(),
                "salis": countries.most_common(1)[0][0] if countries else None,
                "kiekis": len(items),
                "nuotraukos": [p["id"] for p in items],
            }
        )
    for p in dated:
        p.pop("_dt", None)
    trips_out.sort(key=lambda t: t["pradzia"], reverse=True)
    return trips_out
