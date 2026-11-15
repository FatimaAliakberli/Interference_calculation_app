#!/usr/bin/env python
"""
tools/fetch_web_assets.py - run ONCE, with internet, to fill ui/static/web_assets/ with the
JS/CSS/font files Folium maps normally load from public CDNs. After that the app is offline.

    python tools/fetch_web_assets.py            # download everything
    python tools/fetch_web_assets.py --force    # re-download existing files

Resulting layout (desktop_app serves the folder at /web_assets):
    leaflet/leaflet.js, leaflet.css, images/*.png
    jquery/jquery.min.js
    bootstrap/bootstrap.bundle.min.js, bootstrap.min.css
    fontawesome/css/all.min.css, fontawesome/webfonts/*          (css refers to ../webfonts)
    awesome-markers/leaflet.awesome-markers.{js,css}, leaflet.awesome.rotate.min.css, images/*
    glyphicons/bootstrap-glyphicons.css                          (empty stub, see below)

Versions match what Folium 0.15-0.20 links. Check your own with:
    python -c "import folium; print(folium.Map.default_js, folium.Map.default_css)"
If yours differ, edit the URLs below; the rewriter in core/map_visualization.py matches by file
name, and the app logs any CDN URL that is still remote.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "ui" / "static" / "web_assets"

LEAFLET = "https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist"
BOOTSTRAP = "https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist"
FA = "https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0"
AWESOME = "https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2"

FILES = {
    "leaflet/leaflet.js": f"{LEAFLET}/leaflet.js",
    "leaflet/leaflet.css": f"{LEAFLET}/leaflet.css",
    "leaflet/images/layers.png": f"{LEAFLET}/images/layers.png",
    "leaflet/images/layers-2x.png": f"{LEAFLET}/images/layers-2x.png",
    "leaflet/images/marker-icon.png": f"{LEAFLET}/images/marker-icon.png",
    "leaflet/images/marker-icon-2x.png": f"{LEAFLET}/images/marker-icon-2x.png",
    "leaflet/images/marker-shadow.png": f"{LEAFLET}/images/marker-shadow.png",
    "jquery/jquery.min.js": "https://code.jquery.com/jquery-3.7.1.min.js",
    "bootstrap/bootstrap.bundle.min.js": f"{BOOTSTRAP}/js/bootstrap.bundle.min.js",
    "bootstrap/bootstrap.min.css": f"{BOOTSTRAP}/css/bootstrap.min.css",
    "fontawesome/css/all.min.css": f"{FA}/css/all.min.css",
    "fontawesome/webfonts/fa-solid-900.woff2": f"{FA}/webfonts/fa-solid-900.woff2",
    "fontawesome/webfonts/fa-regular-400.woff2": f"{FA}/webfonts/fa-regular-400.woff2",
    "fontawesome/webfonts/fa-brands-400.woff2": f"{FA}/webfonts/fa-brands-400.woff2",
    "fontawesome/webfonts/fa-solid-900.ttf": f"{FA}/webfonts/fa-solid-900.ttf",
    "fontawesome/webfonts/fa-regular-400.ttf": f"{FA}/webfonts/fa-regular-400.ttf",
    "fontawesome/webfonts/fa-brands-400.ttf": f"{FA}/webfonts/fa-brands-400.ttf",
    "awesome-markers/leaflet.awesome-markers.js": f"{AWESOME}/leaflet.awesome-markers.js",
    "awesome-markers/leaflet.awesome-markers.css": f"{AWESOME}/leaflet.awesome-markers.css",
    "awesome-markers/images/markers-soft.png": f"{AWESOME}/images/markers-soft.png",
    "awesome-markers/images/markers-soft@2x.png": f"{AWESOME}/images/markers-soft@2x.png",
    "awesome-markers/images/markers-shadow.png": f"{AWESOME}/images/markers-shadow.png",
    "awesome-markers/images/markers-shadow@2x.png": f"{AWESOME}/images/markers-shadow@2x.png",
    "awesome-markers/leaflet.awesome.rotate.min.css":
        "https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css",
}

# Folium also links Bootstrap-3 "glyphicons". The app's only marker uses a Font Awesome icon
# (prefix="fa"), so an empty stub keeps the page free of remote requests.
STUBS = {
    "glyphicons/bootstrap-glyphicons.css": "/* intentionally empty: glyphicons are not used (Font Awesome icons only) */\n",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    force = ap.parse_args().force
    failed = []
    for rel, url in FILES.items():
        target = DEST / rel
        if target.is_file() and target.stat().st_size and not force:
            print(f"skip   {rel}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                target.write_bytes(resp.read())
            print(f"ok     {rel}")
        except Exception as exc:  # noqa: BLE001
            failed.append((rel, url, exc))
            print(f"FAILED {rel}: {exc}")
    for rel, text in STUBS.items():
        target = DEST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    if failed:
        print(f"\n{len(failed)} file(s) failed; re-run when online.", file=sys.stderr)
        return 1
    print(f"\nDone: {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())