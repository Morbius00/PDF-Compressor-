"""Merge PDFs without re-encoding anything: pages are copied as they are, so text stays selectable,
links keep working and images are untouched. Each source file can contribute all of its pages or a
selection ("1-3, 5"), and every file gets a bookmark so the merged document is easy to navigate."""
from __future__ import annotations

import base64
import re

import pymupdf

pymupdf.TOOLS.mupdf_display_errors(False)


class MergeError(Exception):
    """Problem the user can fix (bad range, locked file, not a PDF)."""


def _open(data: bytes, name: str) -> pymupdf.Document:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise MergeError(f"{name} doesn't look like a valid PDF.") from e
    if doc.needs_pass or doc.is_encrypted:
        doc.close()
        raise MergeError(f"{name} is password protected. Remove the password first.")
    if doc.page_count == 0:
        doc.close()
        raise MergeError(f"{name} has no pages.")
    return doc


def inspect(data: bytes, name: str = "This file") -> dict:
    """Page count and a small first-page thumbnail for the file list."""
    doc = _open(data, name)
    try:
        page = doc[0]
        scale = 200 / max(page.rect.width, 1)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        thumb = "data:image/jpeg;base64," + base64.b64encode(pix.tobytes("jpeg", jpg_quality=72)).decode()
        return {"pages": doc.page_count, "bytes": len(data), "thumb": thumb,
                "width": round(page.rect.width), "height": round(page.rect.height)}
    finally:
        doc.close()


def parse_pages(spec: str | None, count: int, name: str) -> list[int]:
    """'all' / '' / '1-3, 5, 8-10' -> 0-based page numbers in the order given."""
    spec = (spec or "").strip().lower()
    if spec in ("", "all", "*"):
        return list(range(count))
    pages: list[int] = []
    for part in re.split(r"\s*,\s*", spec):
        m = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", part)
        if not m:
            raise MergeError(f"{name}: “{part}” isn't a page or range. Use something like 1-3, 5.")
        a = int(m.group(1))
        b = int(m.group(2) or a)
        if a < 1 or b < 1:
            raise MergeError(f"{name}: pages start at 1.")
        if a > b:
            raise MergeError(f"{name}: the range {a}-{b} runs backwards.")
        if b > count:
            raise MergeError(f"{name}: page {b} doesn't exist (it has {count} page{'s' if count != 1 else ''}).")
        pages.extend(range(a - 1, b))
    return pages


def _runs(pages: list[int]) -> list[tuple[int, int]]:
    """Group consecutive ascending pages so they can be copied in one call."""
    runs: list[list[int]] = []
    for p in pages:
        if runs and p == runs[-1][1] + 1:
            runs[-1][1] = p
        else:
            runs.append([p, p])
    return [(a, b) for a, b in runs]


def _stem(name: str) -> str:
    return re.sub(r"\.pdf$", "", name, flags=re.I).strip() or "Document"


def merge(items: list[tuple[str, bytes, str | None]], bookmarks: bool = True, title: str = "") -> tuple[bytes, dict]:
    """items: [(file name, file bytes, page spec)] in the order they should appear."""
    if len(items) < 2:
        raise MergeError("Add at least two PDFs to merge.")
    out = pymupdf.open()
    toc: list[list] = []
    used: list[dict] = []
    try:
        for name, data, spec in items:
            src = _open(data, name)
            try:
                pages = parse_pages(spec, src.page_count, name)
                start = out.page_count
                first_dest: dict[int, int] = {}
                for a, b in _runs(pages):
                    base = out.page_count
                    out.insert_pdf(src, from_page=a, to_page=b, links=True, annots=True)
                    for off, srcp in enumerate(range(a, b + 1)):
                        first_dest.setdefault(srcp, base + off)

                if bookmarks:
                    toc.append([1, _stem(name), start + 1])
                shift = 1 if bookmarks else 0
                for entry in src.get_toc(simple=True):
                    lvl, ttl, pg = entry[0], entry[1], entry[2]
                    dest = first_dest.get(pg - 1)
                    if dest is None:
                        continue  # that bookmark points at a page we didn't copy
                    toc.append([lvl + shift, ttl, dest + 1])
                used.append({"name": name, "pages": len(pages)})
            finally:
                src.close()

        # bookmark levels must start at 1 and rise by at most one at a time
        fixed, prev = [], 0
        for lvl, ttl, pg in toc:
            lvl = max(1, min(lvl, prev + 1))
            fixed.append([lvl, ttl, pg])
            prev = lvl
        if fixed:
            out.set_toc(fixed)
        if title:
            out.set_metadata({"title": title})
        data = out.tobytes(garbage=4, deflate=True, deflate_fonts=True, use_objstms=1)
        return data, {"pages": out.page_count, "bytes": len(data), "files": used}
    finally:
        out.close()
