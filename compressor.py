"""PDF shrinker that never rasterises pages.

Text, vector art and link annotations stay live. The size you ask for drives how much is changed: the
gentlest change that fits is used, so the result lands close to your size at the best quality that size allows.

  1. lossless : subset fonts, drop junk, dedupe objects, recompress streams (always)
  2. vectors  : coordinate rounding, then, for text drawn as glyph outlines (a common Figma export),
                converting as little of it as needed into a shared embedded font; the rest keeps its
                exact original outlines
  3. images   : only if still over: the highest image quality that fits, starting from the original images
  4. verify   : extracted text and every link must match the original, and vector rewrites must look
                identical when rendered (missing or extra ink is rejected)
"""
from __future__ import annotations

import bisect
import io
import math
import struct
import zlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pymupdf
from PIL import Image

import vectorpack

KB = 1024

NATIVE = 100_000  # "dpi cap" that means: never downsample

# (max effective image dpi, JPEG quality, vector decimals or None), gentlest first. Sizes only shrink downwards.
LADDER = [
    (NATIVE, 98, None), (NATIVE, 96, None), (900, 95, None), (750, 94, None), (650, 94, None), (560, 93, None),
    (480, 93, None), (420, 92, None), (370, 92, None), (330, 91, None), (300, 91, None), (280, 90, None),
    (260, 89, None), (240, 88, None), (220, 87, None), (200, 86, None), (185, 84, None), (170, 82, None),
    (155, 80, None), (142, 78, None), (130, 75, None), (120, 72, None), (110, 68, None), (100, 64, 2),
    (90, 60, 2), (80, 55, 2), (72, 50, 2), (60, 45, 2), (50, 40, 2),
]
FALLBACK_STEP = 14  # (220 dpi, JPEG 87): what we return when the target can't be reached
FILL_BAND = 0.96   # a partial vector conversion is good enough once it lands this close under the target

pymupdf.TOOLS.mupdf_display_errors(False)


class PDFError(Exception):
    """Problem with the input file that the user can act on."""


# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------

def _fingerprint(doc: pymupdf.Document) -> dict:
    pages = []
    for page in doc:
        text = "".join(page.get_text("text").split())  # spacing may legitimately be repaired
        links = sorted(
            (
                l["kind"],
                str(l.get("uri") or l.get("page") or l.get("nameddest") or l.get("file") or ""),
                round(l["from"].x0, 1), round(l["from"].y0, 1),
                round(l["from"].x1, 1), round(l["from"].y1, 1),
            )
            for l in page.get_links()
        )
        pages.append({"size": (round(page.rect.width, 1), round(page.rect.height, 1)),
                      "text": text, "links": links})
    return {"pages": pages, "toc": len(doc.get_toc())}


def _diff(a: dict, b: dict) -> str | None:
    """Human-readable first difference, or None when text/links/pages all match."""
    if len(a["pages"]) != len(b["pages"]):
        return "page count changed"
    if a["toc"] != b["toc"]:
        return "bookmarks changed"
    for i, (pa, pb) in enumerate(zip(a["pages"], b["pages"]), 1):
        if pa["size"] != pb["size"]:
            return f"page {i} size changed"
        if pa["text"] != pb["text"]:
            return f"page {i} text changed"
        if pa["links"] != pb["links"]:
            return f"page {i} links changed"
    return None


VERIFY_DPI = 144
MIN_BLOB_PX = 14  # a missing/extra ink blob at least this large (after anti-alias slack) is a defect


def _visual_defects(orig: bytes, new: bytes, max_pages: int = 24) -> str | None:
    """Render both documents and look for ink that is missing from, or extra in, the new one.
    One pixel of slack absorbs anti-aliasing differences; anything bigger is reported."""
    from scipy import ndimage as ndi

    a, b = pymupdf.open(stream=orig, filetype="pdf"), pymupdf.open(stream=new, filetype="pdf")
    try:
        grow = np.ones((3, 3), bool)
        n = a.page_count
        pages = range(n) if n <= max_pages else sorted({round(i * (n - 1) / (max_pages - 1)) for i in range(max_pages)})
        for pno in pages:
            pa = a[pno].get_pixmap(dpi=VERIFY_DPI, colorspace=pymupdf.csGRAY, alpha=False)
            pb = b[pno].get_pixmap(dpi=VERIFY_DPI, colorspace=pymupdf.csGRAY, alpha=False)
            if (pa.width, pa.height) != (pb.width, pb.height):
                return f"page {pno + 1} renders at a different size"
            ia = np.frombuffer(pa.samples, np.uint8).reshape(pa.height, pa.width) < 150
            ib = np.frombuffer(pb.samples, np.uint8).reshape(pb.height, pb.width) < 150
            for name, mask in (("missing", ia & ~ndi.binary_dilation(ib, grow)),
                               ("extra", ib & ~ndi.binary_dilation(ia, grow))):
                if not mask.any():
                    continue
                lab, n = ndi.label(mask, structure=np.ones((3, 3)))
                sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
                if (sizes >= MIN_BLOB_PX).any():
                    return f"page {pno + 1} has {name} ink after rewriting"
    finally:
        a.close()
        b.close()
    return None


# ----------------------------------------------------------------------------
# Stage 1: lossless
# ----------------------------------------------------------------------------

def _save(doc: pymupdf.Document) -> bytes:
    return doc.tobytes(garbage=4, deflate=True, deflate_fonts=True, deflate_images=True,
                       use_objstms=1, compression_effort=100)


def _recompress_flate(doc: pymupdf.Document) -> None:
    """Re-deflate plain Flate streams (content streams, fonts, CMaps) at zlib level 9."""
    for xref in range(1, doc.xref_length()):
        try:
            if not doc.xref_is_stream(xref):
                continue
            kind, filt = doc.xref_get_key(xref, "Filter")
            if kind != "name" or filt != "/FlateDecode":
                continue
            if doc.xref_get_key(xref, "DecodeParms")[0] != "null":
                continue
            if doc.xref_get_key(xref, "Subtype")[1] == "/Image":
                continue
            raw = doc.xref_stream_raw(xref)
            if len(raw) < 512:
                continue
            better = zlib.compress(doc.xref_stream(xref), 9)
            if len(better) < len(raw):
                doc.update_stream(xref, better, compress=0)
                doc.xref_set_key(xref, "Filter", "/FlateDecode")
        except Exception:
            continue  # leave that stream as it was


_DROP_TABLES = ("fpgm", "prep", "cvt ", "hdmx", "VDMX", "LTSH", "PCLT", "DSIG", "kern", "GSUB", "GPOS",
                "GDEF", "gasp", "meta")


def _trim_truetype(doc: pymupdf.Document) -> int:
    """Strip hinting programs and layout tables from embedded TrueType fonts (a PDF has already
    positioned every glyph and viewers render unhinted). Returns bytes saved."""
    from fontTools.ttLib import TTFont

    saved = 0
    for desc in range(1, doc.xref_length()):
        ff = _key(doc, desc, "FontFile2")
        if not ff:
            continue
        fx = int(ff.split()[0])
        try:
            raw = doc.xref_stream(fx)
            old = len(doc.xref_stream_raw(fx))
            font = TTFont(io.BytesIO(raw), recalcBBoxes=False)
            if "glyf" in font:
                glyf = font["glyf"]
                for name in font.getGlyphOrder():
                    g = glyf[name]
                    if g.numberOfContours != 0:
                        g.removeHinting()
            for t in _DROP_TABLES:
                if t in font:
                    del font[t]
            if "name" in font:
                font["name"].names = [n for n in font["name"].names if n.nameID in (1, 2, 4, 6)]
            maxp = font["maxp"]
            maxp.maxSizeOfInstructions = 0
            for a in ("maxFunctionDefs", "maxStorage", "maxStackElements", "maxTwilightPoints"):
                if hasattr(maxp, a):
                    setattr(maxp, a, 0)
            buf = io.BytesIO()
            font.save(buf)
            new = buf.getvalue()
            TTFont(io.BytesIO(new))  # must still parse
            z = zlib.compress(new, 9)
            if len(z) < old:
                doc.update_stream(fx, z, compress=0)
                doc.xref_set_key(fx, "Filter", "/FlateDecode")
                doc.xref_set_key(fx, "Length1", str(len(new)))
                saved += old - len(z)
        except Exception:
            continue  # leave that font as it was
    return saved


def _lossless(data: bytes, subset_fonts: bool, trim_fonts: bool = True) -> bytes:
    doc = pymupdf.open(stream=data, filetype="pdf")
    if subset_fonts:
        try:
            doc.subset_fonts()
        except Exception:
            pass
    if trim_fonts:
        _trim_truetype(doc)
    try:
        doc.del_xml_metadata()
    except Exception:
        pass
    _recompress_flate(doc)
    out = _save(doc)
    doc.close()
    return out


# ----------------------------------------------------------------------------
# Stage 2: vectors
# ----------------------------------------------------------------------------

def _verify(original: bytes, cand: bytes, want: dict) -> str | None:
    """Why this candidate must not be used, or None. Text and links must match; looks must match."""
    chk = pymupdf.open(stream=cand, filetype="pdf")
    try:
        problem = _diff(want, _fingerprint(chk))
    finally:
        chk.close()
    return problem or _visual_defects(original, cand)


def _round_only(base: bytes, nd: int) -> bytes | None:
    doc = pymupdf.open(stream=base, filetype="pdf")
    try:
        return _save(doc) if vectorpack.round_content_numbers(doc, nd) else None
    except Exception:
        return None
    finally:
        doc.close()


def _vector_phase(base: bytes, original: bytes, want: dict, target: int) -> tuple[bytes, list[str], bool]:
    """Least-invasive vector change that fits the target. -> (bytes, notes, fits)."""
    # a) coordinate rounding: invisible at 3+ decimals, so take the gentlest level that fits
    rounded = {nd: _round_only(base, nd) for nd in (5, 4, 3)}
    r3 = rounded[3]
    useful = r3 is not None and len(r3) < len(base)
    if useful:
        for nd in (5, 4, 3):
            cand = rounded[nd]
            if cand is not None and len(cand) <= target and _verify(original, cand, want) is None:
                return cand, [f"vector coordinates rounded to {nd} decimals (no visible change)"], True
    level0, level0_notes = base, []

    # b) text drawn as outlines: convert only as much of it as the target needs
    try:
        adoc = pymupdf.open(stream=base, filetype="pdf")
        try:
            an = vectorpack.analyze_outline_text(adoc)
        finally:
            adoc.close()
    except Exception:  # not that kind of file, or not safe to touch
        an = None
    if an is None or not an.order:
        if useful and _verify(original, r3, want) is None:
            level0, level0_notes = r3, ["vector coordinates rounded to 3 decimals (no visible change)"]
        return level0, level0_notes, len(level0) <= target

    total = len(an.order)

    def make(k: int) -> bytes:
        doc = pymupdf.open(stream=base, filetype="pdf")
        try:
            vectorpack.apply_outline_text(doc, an, k, 3)
            return _save(doc)
        finally:
            doc.close()

    full = make(total)
    k, out = total, full
    if len(full) <= target:
        # size shrinks as more text is converted: interpolate, then tighten, to land just under the target
        s_lo = max(len(r3) if useful else len(base), int(target * 1.02))
        lo, hi, s_hi, best = 0, total, len(full), (total, full)
        for _ in range(7):
            if s_hi >= FILL_BAND * target or hi - lo <= 1 or s_lo <= s_hi:
                break
            frac = (s_lo - 0.98 * target) / (s_lo - s_hi)
            x = an.cum[lo] + frac * (an.cum[hi] - an.cum[lo])
            probe = min(max(bisect.bisect_left(an.cum, x), lo + 1), hi - 1)
            cand = make(probe)
            if len(cand) <= target:
                hi, s_hi, best = probe, len(cand), (probe, cand)
            else:
                lo, s_lo = probe, len(cand)
        k, out = best

    for kk, cand in ([(k, out)] + ([(total, full)] if k != total else [])):
        if _verify(original, cand, want) is None:
            pct = round(100 * an.cum[kk] / an.cum[-1])
            if kk == total:
                note = (f"rebuilt {an.cum[-1]:,} outlined letters as {len(an.glyphs)} shared glyphs "
                        f"in one embedded font")
            else:
                note = (f"rebuilt {pct}% of the lettering (smallest text first) as a shared font; "
                        f"larger text keeps its exact original outlines")
            return cand, [note], len(cand) <= target
    if useful and _verify(original, r3, want) is None:
        level0, level0_notes = r3, ["vector coordinates rounded to 3 decimals (no visible change)"]
    return level0, level0_notes, len(level0) <= target


# ----------------------------------------------------------------------------
# Stage 3: images (+ vector precision)
# ----------------------------------------------------------------------------

@dataclass
class _Job:
    xref: int
    is_mask: bool
    disp_w: float  # largest on-page width in points (0 = never seen on a page)
    disp_h: float
    orig_len: int


_SIMPLE_CS = ("DeviceRGB", "DeviceGray", "CalRGB", "CalGray", "ICCBased(RGB", "ICCBased(Gray")


def _key(doc, xref, name):
    t, v = doc.xref_get_key(xref, name)
    return None if t == "null" else v


def _eligible(doc, xref) -> bool:
    if _key(doc, xref, "ImageMask") == "true":
        return False
    for k in ("Mask", "Decode", "SMaskInData"):
        if _key(doc, xref, k):
            return False
    smask = _key(doc, xref, "SMask")
    if smask:
        sx = int(smask.split()[0])
        if _key(doc, sx, "Matte") or _key(doc, sx, "Decode"):
            return False
    return True


def _collect_jobs(doc: pymupdf.Document) -> list[_Job]:
    sizes: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for page in doc:
        for im in page.get_image_info(xrefs=True):
            x = im.get("xref", 0)
            if not x:
                continue
            a, b, c, d = im["transform"][:4]
            sz = sizes[x]
            sz[0] = max(sz[0], math.hypot(a, b))
            sz[1] = max(sz[1], math.hypot(c, d))

    jobs: dict[int, _Job] = {}
    for xref in range(1, doc.xref_length()):
        if _key(doc, xref, "Subtype") != "/Image" or not doc.xref_is_stream(xref):
            continue
        if not _eligible(doc, xref):
            continue
        w, h = sizes.get(xref, (0.0, 0.0))
        jobs[xref] = _Job(xref, False, w, h, len(doc.xref_stream_raw(xref)))

    # soft masks follow the display size of the image(s) that use them
    for xref, job in list(jobs.items()):
        smask = _key(doc, xref, "SMask")
        if not smask:
            continue
        sx = int(smask.split()[0])
        m = jobs.get(sx) or _Job(sx, True, 0.0, 0.0, len(doc.xref_stream_raw(sx)))
        m.is_mask = True
        m.disp_w = max(m.disp_w, job.disp_w)
        m.disp_h = max(m.disp_h, job.disp_h)
        jobs[sx] = m
    return list(jobs.values())


def _to_pil(doc, job: _Job) -> tuple[Image.Image, str | None]:
    """Decode to PIL. Second value is a replacement /ColorSpace (None = keep original)."""
    pix = pymupdf.Pixmap(doc, job.xref)
    if pix.alpha:  # alpha is handled through the separate SMask object
        pix = pymupdf.Pixmap(pix, 0)
    cs_name = pix.colorspace.name if pix.colorspace else ""
    new_cs = None
    if job.is_mask or pix.n == 1:
        mode = "L"
        if not job.is_mask and not any(cs_name.startswith(s) for s in _SIMPLE_CS):
            new_cs = "/DeviceGray"
    else:
        if pix.n != 3 or not any(cs_name.startswith(s) for s in _SIMPLE_CS):
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            new_cs = "/DeviceRGB"
        mode = "RGB"
    samples = (pix.samples if pix.stride == pix.width * pix.n else
               b"".join(pix.samples[r * pix.stride:r * pix.stride + pix.width * pix.n] for r in range(pix.height)))
    return Image.frombytes(mode, (pix.width, pix.height), samples), new_cs


def _flate_png(im: Image.Image) -> tuple[bytes, dict]:
    """Lossless Flate with PNG row filters, reusing Pillow's PNG encoder."""
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True, compress_level=9)
    png = buf.getvalue()
    pos, idat = 8, []
    while pos < len(png):
        n, typ = struct.unpack(">I4s", png[pos:pos + 8])
        if typ == b"IDAT":
            idat.append(png[pos + 8:pos + 8 + n])
        pos += 12 + n
    parms = {"Predictor": 15, "Colors": 3 if im.mode == "RGB" else 1,
             "BitsPerComponent": 8, "Columns": im.width}
    return b"".join(idat), parms


def _jpeg(im: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True,
            subsampling=(2 if quality <= 85 else 0) if im.mode == "RGB" else 0)
    return buf.getvalue()


def _encode(im: Image.Image, is_mask: bool, quality: int):
    """Return (data, filter, decodeparms|None), picking the smallest suitable encoding."""
    flat, parms = _flate_png(im)
    if is_mask:
        # smooth soft-edged masks can go through JPEG, hard-edged ones stay lossless
        if len(flat) > 8 * KB:
            jpg = _jpeg(im, max(quality, 75))
            if len(jpg) < 0.6 * len(flat):
                return jpg, "/DCTDecode", None
        return flat, "/FlateDecode", parms
    jpg = _jpeg(im, quality)
    # flat graphics / screenshots: lossless Flate beats JPEG and has no artefacts
    if im.getcolors(maxcolors=4096) is not None or len(flat) < len(jpg):
        return flat, "/FlateDecode", parms
    return jpg, "/DCTDecode", None


def _apply_step(base: bytes, jobs: list[_Job], cache: dict, dpi: int, quality: int, nd: int | None) -> bytes:
    doc = pymupdf.open(stream=base, filetype="pdf")
    for job in jobs:
        try:
            if job.xref not in cache:
                cache[job.xref] = _to_pil(doc, job)
            im, new_cs = cache[job.xref]

            f = 1.0
            if job.disp_w > 0 and job.disp_h > 0:
                f = min(1.0, max(dpi * job.disp_w / 72 / im.width, dpi * job.disp_h / 72 / im.height))
            elif max(im.size) > 1600:  # never drawn on a page we can see: cap by pixel size
                f = 1600 / max(im.size)
            out = im
            if f < 0.999:
                out = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))),
                                Image.Resampling.LANCZOS)

            data, filt, parms = _encode(out, job.is_mask, quality)
            if len(data) >= job.orig_len:
                continue  # never make an image bigger than it already is

            doc.update_stream(job.xref, data, compress=0)
            doc.xref_set_key(job.xref, "Filter", filt)
            doc.xref_set_key(job.xref, "DecodeParms",
                             "null" if parms is None else
                             "<<" + "".join(f"/{k} {v}" for k, v in parms.items()) + ">>")
            doc.xref_set_key(job.xref, "Width", str(out.width))
            doc.xref_set_key(job.xref, "Height", str(out.height))
            doc.xref_set_key(job.xref, "BitsPerComponent", "8")
            if new_cs:
                doc.xref_set_key(job.xref, "ColorSpace", new_cs)
        except Exception:
            continue  # unusual image: leave it untouched
    if nd is not None:
        try:
            vectorpack.round_content_numbers(doc, nd)
        except Exception:
            pass
    out = _save(doc)
    doc.close()
    return out


def _size_breakdown(data: bytes) -> dict:
    doc = pymupdf.open(stream=data, filetype="pdf")
    images = fonts = 0
    font_files = set()
    for xref in range(1, doc.xref_length()):
        for k in ("FontFile", "FontFile2", "FontFile3"):
            v = _key(doc, xref, k)
            if v:
                font_files.add(int(v.split()[0]))
    for xref in range(1, doc.xref_length()):
        if not doc.xref_is_stream(xref):
            continue
        n = len(doc.xref_stream_raw(xref))
        if _key(doc, xref, "Subtype") == "/Image":
            images += n
        elif xref in font_files:
            fonts += n
    doc.close()
    return {"images": images, "fonts": fonts, "other": max(0, len(data) - images - fonts)}


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def compress(data: bytes, target_bytes: int = 100 * KB) -> tuple[bytes, dict]:
    try:
        src = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise PDFError("That file doesn't look like a valid PDF.") from e
    if src.needs_pass or src.is_encrypted:
        raise PDFError("This PDF is password protected. Remove the password first.")
    if src.page_count == 0:
        raise PDFError("This PDF has no pages.")
    want = _fingerprint(src)
    src.close()

    steps: list[str] = []

    # 1. lossless; fall back to gentler variants if anything about text, links or looks changed
    base, problem = None, None
    for subset, trim in ((True, True), (True, False), (False, False)):
        cand = _lossless(data, subset, trim)
        chk = pymupdf.open(stream=cand, filetype="pdf")
        problem = _diff(want, _fingerprint(chk))
        chk.close()
        if problem is None and (trim or subset):
            problem = _visual_defects(data, cand)
        if problem is None:
            base = cand
            break
    if base is None:
        raise PDFError(f"Couldn't shrink this PDF safely ({problem}); original left untouched.")
    if len(base) >= len(data):
        base = data  # nothing gained, keep the original bytes
    steps.append("cleaned up structure and recompressed streams")

    # 2. vectors: the gentlest change that fits the size you asked for
    final, step, lossy = base, None, False
    if len(base) > target_bytes:
        final, notes, fits = _vector_phase(base, data, want, target_bytes)
        steps += notes

        # 3. images: the highest quality that fits (only when vectors alone couldn't get there)
        if not fits:
            doc = pymupdf.open(stream=final, filetype="pdf")
            jobs = _collect_jobs(doc)
            doc.close()
            cache: dict = {}
            trials: dict[int, bytes] = {}

            def trial(i: int) -> bytes:
                if i not in trials:
                    trials[i] = _apply_step(final, jobs, cache, *LADDER[i])
                return trials[i]

            lo, hi = 0, len(LADDER) - 1
            if len(trial(hi)) < len(final):
                if len(trial(hi)) <= target_bytes:
                    while lo < hi:  # smallest index (= best quality) that still fits
                        mid = (lo + hi) // 2
                        if len(trial(mid)) <= target_bytes:
                            hi = mid
                        else:
                            lo = mid + 1
                    pick = lo
                else:
                    pick = FALLBACK_STEP  # out of reach: keep a good-looking shrink rather than wreck quality
                cand = trial(pick)
                if len(cand) < len(final):
                    chk = pymupdf.open(stream=cand, filetype="pdf")
                    problem = _diff(want, _fingerprint(chk))
                    chk.close()
                    if problem:
                        raise PDFError(f"Image optimisation altered the document ({problem}).")
                    final, step = cand, LADDER[pick]
                    lossy = bool(jobs) or step[2] is not None
                    if jobs:
                        steps.append("images re-encoded at their original resolution (JPEG quality %d)" % step[1]
                                     if step[0] == NATIVE else
                                     f"images resized to {step[0]} dpi and re-encoded (JPEG quality {step[1]})")
                    if step[2] is not None:
                        steps.append("vector coordinates rounded to 0.01 pt")

    out = pymupdf.open(stream=final, filetype="pdf")
    fp = _fingerprint(out)
    fonts = {f[3] for p in out for f in p.get_fonts()}
    out.close()
    info = {
        "original_bytes": len(data),
        "final_bytes": len(final),
        "target_bytes": target_bytes,
        "reached_target": len(final) <= target_bytes,
        "pages": len(fp["pages"]),
        "text_chars": sum(len(p["text"]) for p in fp["pages"]),
        "links": sum(len(p["links"]) for p in fp["pages"]),
        "fonts": len(fonts),
        "steps": steps,
        "image_dpi": step[0] if step and step[0] != NATIVE else None,
        "image_quality": step[1] if step else None,
        "lossy": lossy,
        "breakdown": _size_breakdown(final),
    }
    return final, info
