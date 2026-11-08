"""Dailės studija – lokalus serveris (http://localhost:8765). Niekas nesiunčiama į internetą."""
from __future__ import annotations

import io
import json
import mimetypes
import os
import subprocess
import sys
import threading
import traceback
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import guide as G
from . import imaging, sources

PORT = int(os.environ.get("STUDIJA_PORT", "8765"))
WEB = Path(__file__).parent / "web"
DATA = Path.home() / "Library" / "Application Support" / "DailesStudija"
THUMBS = DATA / "thumbs"
RENDERS = DATA / "renders"
STATE_FILE = DATA / "state.json"
DATA.mkdir(parents=True, exist_ok=True)

LOCK = threading.Lock()
STATE = {"photos": {}, "trips": [], "metrics": {}, "selected": {}, "source": None, "settings": {}}
PROGRESS = {"running": False, "text": "", "pct": 0, "error": None}


def load_state():
    if STATE_FILE.exists():
        try:
            STATE.update(json.loads(STATE_FILE.read_text()))
        except Exception:
            pass


def save_state():
    with LOCK:
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(STATE, ensure_ascii=False))
        tmp.replace(STATE_FILE)


def progress(text, pct):
    PROGRESS.update(text=text, pct=round(pct, 1))


# ------------------------------------------------------------------ vertinimas
def total_score(photo: dict, m: dict) -> float:
    s = 0.45 * m["tapybiskumas"] + 0.35 * m["kokybe"]
    a = photo.get("apple_score")
    s += 0.20 * (max(0.0, min(1.0, a)) * 100 if a is not None else (m["tapybiskumas"] + m["kokybe"]) / 2)
    if photo.get("favorite"):
        s += 8
    if photo.get("selfie"):
        s -= 25
    else:
        # kelionių peizažui žmonės kadre – minusas; kuo didesnis veidas, tuo didesnis
        fs = photo.get("face_size") or 0
        s -= min(28, fs * 220)
        if photo.get("faces", 0) >= 3:
            s -= 8
        elif photo.get("faces", 0) >= 1 and not fs:
            s -= 6
    if m["ekspozicija"] < 0.25:
        s -= 10
    return round(max(0, min(100, s)), 1)


def run_scan(source: str, folder: str | None, mode: str, since_year: int | None, per_trip: int):
    try:
        PROGRESS.update(running=True, error=None, pct=0, text="Pradedama…")
        if source == "photos":
            photos = sources.scan_photos(progress, since_year)
        else:
            photos = sources.scan_folder(folder, progress, since_year)
        if not photos:
            raise ValueError("Nerasta nė vienos nuotraukos.")
        progress("Ieškoma kelionių…", 26)
        trips = sources.group_trips(photos, mode=mode)
        if not trips and mode == "auto":
            trips = sources.group_trips(photos, mode="folder" if source == "folder" else "album")
        if not trips:
            raise ValueError("Nepavyko atpažinti kelionių. Pabandykite grupavimą „pagal aplankus/albumus“.")
        pmap = {p["id"]: p for p in photos}

        # kandidatai analizei
        todo = []
        for t in trips:
            ids = t["nuotraukos"]
            if any(pmap[i].get("apple_score") is not None for i in ids):
                ranked = sorted(ids, key=lambda i: (pmap[i].get("apple_score") or -1) + (0.3 if pmap[i]["favorite"] else 0),
                                reverse=True)
                cand = ranked[: max(per_trip * 4, 30)]
            else:
                step = max(1, len(ids) // 150)
                cand = ids[::step]
            t["_cand"] = cand
            todo.extend(cand)
        todo = list(dict.fromkeys(todo))
        metrics = STATE.get("metrics", {})
        for n, pid in enumerate(todo):
            if n % 5 == 0:
                progress(f"Vertinamos nuotraukos: {n}/{len(todo)}", 27 + 70 * n / max(len(todo), 1))
            p = pmap[pid]
            key = f"{p['path']}|{os.path.getmtime(p['path']) if os.path.exists(p['path']) else 0}"
            if pid in metrics and metrics[pid].get("_key") == key and (THUMBS / f"{pid}.jpg").exists():
                continue
            try:
                m = imaging.analyze(p["path"])
                m["_key"] = key
                metrics[pid] = m
                imaging.make_thumb(p["path"], THUMBS / f"{pid}.jpg")
            except Exception:
                metrics.pop(pid, None)

        # atrinkimas: balas + dublikatų šalinimas
        for t in trips:
            scored = []
            for pid in t.pop("_cand"):
                m = metrics.get(pid)
                if not m or m.get("dokumentas"):
                    continue
                scored.append((total_score(pmap[pid], m), pid))
            scored.sort(reverse=True)
            best, kept = [], []
            for sc, pid in scored:
                h = metrics[pid]["dhash"]
                d = pmap[pid].get("date")
                dt = datetime.fromisoformat(d) if d else None

                def similar(o):
                    oh, odt = o
                    dist = imaging.hamming(h, oh)
                    if dist <= 10:
                        return True
                    # serija: kadrai per 3 min. ir panaši kompozicija
                    return bool(dt and odt and abs((dt - odt).total_seconds()) < 180 and dist <= 22)

                if any(similar(o) for o in kept):
                    continue
                kept.append((h, dt))
                best.append({"id": pid, "balas": sc})
                if len(best) >= per_trip:
                    break
            t["geriausios"] = best
            t["virselis"] = best[0]["id"] if best else None
        trips = [t for t in trips if t.get("geriausios")]
        with LOCK:
            STATE["photos"] = pmap
            STATE["trips"] = trips
            STATE["metrics"] = metrics
            STATE["source"] = {"tipas": source, "aplankas": folder, "grupavimas": mode}
        save_state()
        progress(f"Baigta! Rasta kelionių: {len(trips)}", 100)
    except Exception as e:
        traceback.print_exc()
        PROGRESS["error"] = str(e)
    finally:
        PROGRESS["running"] = False


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype="application/json; charset=utf-8", cache=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode())

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        try:
            self.route_get()
        except Exception as e:
            traceback.print_exc()
            self._json({"klaida": str(e)}, 500)

    def do_POST(self):
        try:
            self.route_post()
        except Exception as e:
            traceback.print_exc()
            self._json({"klaida": str(e)}, 500)

    def route_get(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        parts = [x for x in u.path.split("/") if x]

        if not parts or parts[0] in ("index.html",):
            return self._file(WEB / "index.html")
        if parts[0] == "web":
            f = (WEB / "/".join(parts[1:])).resolve()
            if WEB.resolve() in f.parents and f.exists():
                return self._file(f)
            return self._json({"klaida": "nerasta"}, 404)
        if parts[0] == "api":
            return self.api_get(parts[1:], q)
        if parts[0] == "thumb" and len(parts) == 2:
            f = THUMBS / f"{parts[1]}.jpg"
            if f.exists():
                return self._send(200, f.read_bytes(), "image/jpeg", cache=True)
            p = STATE["photos"].get(parts[1])
            if p:
                imaging.make_thumb(p["path"], f)
                return self._send(200, f.read_bytes(), "image/jpeg", cache=True)
            return self._json({"klaida": "nerasta"}, 404)
        if parts[0] == "render" and len(parts) == 3:
            return self.render(parts[1], parts[2], q)
        return self._json({"klaida": "nerasta"}, 404)

    def _file(self, f: Path):
        ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self._send(200, f.read_bytes(), ctype)

    def _photo_out(self, pid, balas=None):
        p = STATE["photos"].get(pid, {})
        m = STATE["metrics"].get(pid, {})
        return {
            "id": pid,
            "balas": balas,
            "data": p.get("date"),
            "vieta": p.get("place"),
            "failas": p.get("filename"),
            "megstama": p.get("favorite"),
            "kokybe": m.get("kokybe"),
            "tapybiskumas": m.get("tapybiskumas"),
            "scena": m.get("scena"),
            "pasirinkta": pid in STATE["selected"],
        }

    def api_get(self, parts, q):
        if parts == ["busena"]:
            return self._json({
                "photos": sources.photos_status(),
                "progress": PROGRESS,
                "turim": bool(STATE["trips"]),
                "saltinis": STATE.get("source"),
                "pasirinkta": len(STATE["selected"]),
            })
        if parts == ["progress"]:
            return self._json(PROGRESS)
        if parts == ["keliones"]:
            out = []
            for t in STATE["trips"]:
                out.append({k: v for k, v in t.items() if k not in ("nuotraukos", "geriausios")}
                           | {"geriausios": [self._photo_out(b["id"], b["balas"]) for b in t["geriausios"]]})
            return self._json(out)
        if parts == ["aplankas"]:
            r = subprocess.run(["osascript", "-e",
                                'POSIX path of (choose folder with prompt "Pasirinkite aplanką su kelionių nuotraukomis")'],
                               capture_output=True, text=True)
            return self._json({"kelias": r.stdout.strip() or None})
        if len(parts) == 2 and parts[0] == "planas":
            pid = parts[1]
            p = STATE["photos"].get(pid)
            if not p:
                return self._json({"klaida": "Nuotrauka nerasta"}, 404)
            m = STATE["metrics"].get(pid) or imaging.analyze(p["path"])
            medium = q.get("m", "akrilas")
            ci = int(q.get("c", STATE["selected"].get(pid, {}).get("c", 1)))
            g = G.build_guide(p, m, medium, ci)
            g["nuotrauka"] = self._photo_out(pid)
            g["metrikos"] = {k: v for k, v in m.items() if not k.startswith("_") and k != "dhash"}
            return self._json(g)
        if parts == ["pirkiniai"]:
            medium = q.get("m", "akrilas")
            pals, canv, items = [], [], []
            for pid, sel in STATE["selected"].items():
                p = STATE["photos"].get(pid)
                if not p:
                    continue
                img = imaging.load_image(p["path"], max_side=700)
                opts = G.canvas_options(*img.size)
                cv = opts[min(sel.get("c", 1), 2)]
                pal = G.extract_palette(G.crop_to(img, cv["plotis"], cv["aukstis"]))
                pals.append(pal)
                canv.append(cv)
                items.append(self._photo_out(pid) | {"drobe": cv})
            return self._json({"paveikslai": items, "sarasas": G.supplies(pals, canv, medium) if pals else None})
        return self._json({"klaida": "nežinomas kelias"}, 404)

    def render(self, pid, kind, q):
        p = STATE["photos"].get(pid)
        if not p:
            return self._json({"klaida": "nerasta"}, 404)
        ci = int(q.get("c", 1))
        img = imaging.load_image(p["path"], max_side=900)
        cv = G.canvas_options(*img.size)[min(ci, 2)]
        cache = RENDERS / f"{pid}_{kind}_{cv['plotis']}x{cv['aukstis']}.jpg"
        if not cache.exists():
            pal = None
            if kind in ("zonos", "konturai"):
                pal = G.extract_palette(G.crop_to(img, cv["plotis"], cv["aukstis"]))
            out = G.render(kind, p["path"], cv["plotis"], cv["aukstis"], pal)
            cache.parent.mkdir(parents=True, exist_ok=True)
            out.save(cache, "JPEG", quality=88)
        self._send(200, cache.read_bytes(), "image/jpeg", cache=True)

    def route_post(self):
        parts = [x for x in urlparse(self.path).path.split("/") if x]
        body = self._body()
        if parts == ["api", "skenuoti"]:
            if PROGRESS["running"]:
                return self._json({"klaida": "Jau skenuojama"}, 409)
            src = body.get("saltinis", "photos")
            folder = body.get("aplankas")
            if src == "folder" and not folder:
                return self._json({"klaida": "Nurodykite aplanką"}, 400)
            since = int(body["nuo_metu"]) if body.get("nuo_metu") else None
            th = threading.Thread(target=run_scan, args=(src, folder, body.get("grupavimas", "auto"), since,
                                                         int(body.get("kiekis", 12))), daemon=True)
            th.start()
            return self._json({"ok": True})
        if parts == ["api", "pasirinkti"]:
            pid = body["id"]
            with LOCK:
                if body.get("on"):
                    STATE["selected"][pid] = {"c": int(body.get("c", 1))}
                else:
                    STATE["selected"].pop(pid, None)
            save_state()
            return self._json({"ok": True, "pasirinkta": len(STATE["selected"])})
        if parts == ["api", "atidaryti"]:
            p = STATE["photos"].get(body.get("id"))
            if p and os.path.exists(p["path"]):
                subprocess.run(["open", "-R", p["path"]])
            return self._json({"ok": True})
        return self._json({"klaida": "nežinomas kelias"}, 404)


def main():
    load_state()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"\n  🎨 Dailės studija veikia: {url}\n  (Uždarykite šį langą arba spauskite Ctrl+C, kad sustabdytumėte)\n")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
