"""Rendered check for scripts/precheck.mjs: six devices, each page in the light scheme, the core pages also in the dark scheme.

The devices are the ones in humem.ai's docs/design/test-devices.md. Together they put a device on each side of the media
queries in src/styles/global.css.

Fails on horizontal overflow, console errors, a page without exactly one h1, a broken image, an image without an alt
attribute, and a cropped illustration: an `object-fit: cover` image with one ground color in its four corners (an
illustration or a figure, not a photo) whose content the box it is shown in cuts off. Run through `npm run precheck`, or by
hand with the base URL of a running site."""
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

base = sys.argv[1].rstrip("/")


def find_chrome():
    """PLAYWRIGHT_CHROME, else whichever Chromium Playwright installed."""
    if os.environ.get("PLAYWRIGHT_CHROME"):
        return os.environ["PLAYWRIGHT_CHROME"]
    cache = Path.home() / ".cache/ms-playwright"
    installed = sorted(cache.glob("chromium-*/chrome-linux*/chrome"))
    return str(installed[-1]) if installed else str(cache / "chromium/chrome-linux/chrome")


chrome = find_chrome()
if not Path(chrome).exists():
    sys.exit(f"no Chromium at {chrome}; set PLAYWRIGHT_CHROME, or run `uvx --from playwright==1.55.0 playwright install chromium`")

# width, height, device pixel ratio: Small Android, iPhone 14, iPad portrait, Z Fold 7 open, Laptop 1280, Desktop 1920
devices = [(360, 800, 3), (390, 844, 3), (768, 1024, 2), (984, 1092, 2), (1280, 720, 1), (1920, 1080, 1)]

core_pages = ["/", "/blog/", "/contact/", "/privacy-policy/", "/projects/computer-vision/"]
sitemap = urllib.request.urlopen(base + "/sitemap-0.xml").read().decode()
listed = [urllib.parse.urlparse(u).path or "/" for u in re.findall(r"<loc>([^<]+)</loc>", sitemap)]
posts = [p for p in listed if re.match(r"^/\d{4}/\d{2}/\d{2}/", p)]
tags = [p for p in listed if p.startswith("/tags/") and p != "/tags/"][:2]
pages = list(dict.fromkeys(core_pages + posts + tags))

# Runs in the page. Loads every lazy image, then reports broken images, images without alt, and cropped illustrations.
IMAGE_CHECK = """
async () => {
  document.querySelectorAll('img[loading="lazy"]').forEach((i) => { i.loading = 'eager'; });
  await Promise.all([...document.images].map((i) => (i.complete ? 0 : new Promise((r) => {
    i.addEventListener('load', r); i.addEventListener('error', r); setTimeout(r, 5000);
  }))));
  const name = (i) => {
    const u = new URL(i.currentSrc || i.src, location.href);
    return u.pathname.split('/').pop();
  };
  const out = { broken: [], noAlt: [], cropped: [], h1: document.querySelectorAll('h1').length };
  for (const i of document.images) {
    if (!(i.currentSrc || i.src)) continue;
    if (i.naturalWidth === 0) { out.broken.push(name(i)); continue; }
    if (!i.hasAttribute('alt')) out.noAlt.push(name(i));
    const r = i.getBoundingClientRect();
    if (r.width < 60 || r.height < 40 || getComputedStyle(i).objectFit !== 'cover') continue;
    // the part of the image the box shows, as fractions of the image
    const nr = i.naturalWidth / i.naturalHeight, br = r.width / r.height;
    let x0 = 0, y0 = 0, x1 = 1, y1 = 1;
    if (br > nr) { y0 = (1 - nr / br) / 2; y1 = 1 - y0; } else { x0 = (1 - br / nr) / 2; x1 = 1 - x0; }
    // the content of the image: everything that is not the ground color, found on a 160 px copy
    const w = 160, h = Math.max(1, Math.round(w / nr)), c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d'); ctx.drawImage(i, 0, 0, w, h);
    const d = ctx.getImageData(0, 0, w, h).data;
    // an illustration or figure has one ground color in all four corners; a photo does not, and a crop of a photo is fine
    const corner = (x, y) => { const p = (y * w + x) * 4; return [d[p], d[p + 1], d[p + 2]]; };
    const g = corner(0, 0);
    if ([corner(w - 1, 0), corner(0, h - 1), corner(w - 1, h - 1)].some((q) => Math.abs(q[0] - g[0]) + Math.abs(q[1] - g[1]) + Math.abs(q[2] - g[2]) > 12)) continue;
    let a = w, b = h, e = -1, f = -1;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 4;
      if (Math.abs(d[p] - g[0]) + Math.abs(d[p + 1] - g[1]) + Math.abs(d[p + 2] - g[2]) > 60) {
        if (x < a) a = x; if (x > e) e = x; if (y < b) b = y; if (y > f) f = y;
      }
    }
    const tol = 0.004;
    if (a / w < x0 - tol || (e + 1) / w > x1 + tol || b / h < y0 - tol || (f + 1) / h > y1 + tol) {
      out.cropped.push(`${name(i)} ${Math.round(r.width)}x${Math.round(r.height)} shows `
        + `${Math.round((x1 - x0) * 100)}%w ${Math.round((y1 - y0) * 100)}%h, content `
        + `${Math.round((e + 1 - a) / w * 100)}%w ${Math.round((f + 1 - b) / h * 100)}%h`);
    }
  }
  return out;
}
"""

bad = 0
loads = 0
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=chrome)
    for width, height, dpr in devices:
        for scheme in ("light", "dark"):
            context = browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=dpr, color_scheme=scheme)
            for path in pages:
                if scheme == "dark" and path not in core_pages:
                    continue
                page = context.new_page()
                errors = []
                # An embedded YouTube player asks for browser features that the embed's `allow` list does not grant, and
                # Chromium logs that as an error in our page. It is not ours to fix.
                page.on("console", lambda m, e=errors: e.append(m.text) if m.type == "error" and not m.text.startswith("Permissions policy violation") else None)
                page.on("pageerror", lambda x, e=errors: e.append(str(x)))
                page.goto(base + path, wait_until="networkidle")
                scroll_width = page.evaluate("document.documentElement.scrollWidth")
                images = page.evaluate(IMAGE_CHECK) if scheme == "light" else {"broken": [], "noAlt": [], "cropped": [], "h1": 1}
                loads += 1
                problems = []
                if scroll_width != width:
                    problems.append(f"scrollWidth {scroll_width}")
                if errors:
                    problems.append(f"console {errors[:2]}")
                if images["h1"] != 1:
                    problems.append(f"{images['h1']} h1 elements")
                if images["broken"]:
                    problems.append(f"broken image {images['broken']}")
                if images["noAlt"]:
                    problems.append(f"no alt {images['noAlt']}")
                for c in images["cropped"]:
                    problems.append(f"cropped {c}")
                if problems:
                    bad += 1
                    print(f"  FAIL {width}px {scheme} {path}: {'; '.join(problems)}")
                page.close()
            context.close()
    browser.close()
print(f"  {loads} page loads on {len(devices)} devices, {bad} bad")
sys.exit(1 if bad else 0)
