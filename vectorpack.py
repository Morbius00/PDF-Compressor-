"""Vector-side shrinking for PDFs whose text is drawn as outlines.

Figma exports can draw every glyph as a filled path with six-decimal coordinates, so the same
letter is stored hundreds of times, and add a separate *invisible* text layer (Type 3 fonts whose
glyphs are empty) so the text stays selectable.

`transplant_outline_text` pairs every outlined glyph with the text-layer glyph it belongs to,
builds one shared embedded font from the distinct outlines, turns the Type 3 fonts into ordinary
fonts that use it, puts each glyph exactly where its outline was, and deletes the duplicate paths.
Nothing is rasterised; the text layer, its Unicode mapping and the links are kept.

`round_content_numbers` is the generic fallback: it re-serialises content streams with fewer
decimals.
"""
from __future__ import annotations

import math
import re
import zlib
from collections import defaultdict

import pymupdf

# ----------------------------------------------------------------------------
# Content stream tokenising / serialising
# ----------------------------------------------------------------------------

_TOK = re.compile(
    rb"""[ \t\r\n\f\x00]*(?:
        (?P<num>[-+]?(?:\d+\.?\d*|\.\d+))(?=[\s/\[\]()<>%]|$) |
        (?P<name>/[^\s/\[\]()<>%{}]*) |
        (?P<op>[A-Za-z'"*][A-Za-z0-9'"*]*) |
        (?P<arr>\[) | (?P<arre>\]) |
        (?P<dict><<) | (?P<dicte>>>) |
        (?P<hex><[0-9A-Fa-f\s]*>) |
        (?P<str>\() |
        (?P<cmt>%[^\r\n]*)
    )""",
    re.X,
)


class Unsupported(Exception):
    pass


class Name(bytes):
    pass


class Raw(bytes):
    """Operand kept verbatim (string / hex string / dict)."""


def _scan_string(data: bytes, pos: int) -> int:
    depth, i, n = 1, pos, len(data)
    while i < n:
        c = data[i]
        if c == 0x5C:  # backslash
            i += 2
            continue
        if c == 0x28:
            depth += 1
        elif c == 0x29:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise Unsupported("unterminated string")


def parse_ops(data: bytes) -> list[tuple[bytes, list]]:
    """-> [(operator, operands)]. Raises Unsupported for inline images etc."""
    ops: list[tuple[bytes, list]] = []
    stack: list[list] = [[]]  # operand stacks (arrays nest)
    pos, n = 0, len(data)
    match = _TOK.match
    while pos < n:
        m = match(data, pos)
        if not m:
            if data[pos:].strip() == b"":
                break
            raise Unsupported(f"bad token near {data[pos:pos + 20]!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "num":
            t = m.group("num")
            stack[-1].append(float(t) if (b"." in t) else int(t))
        elif kind == "name":
            stack[-1].append(Name(m.group("name")))
        elif kind == "op":
            op = m.group("op")
            if op in (b"BI", b"ID", b"EI"):
                raise Unsupported("inline image")
            if op in (b"true", b"false", b"null"):
                stack[-1].append(Raw(op))
                continue
            if len(stack) != 1:
                raise Unsupported("operator inside array")
            ops.append((op, stack[0]))
            stack[0] = []
        elif kind == "arr":
            stack.append([])
        elif kind == "arre":
            if len(stack) < 2:
                raise Unsupported("stray ]")
            arr = stack.pop()
            stack[-1].append(arr)
        elif kind == "dict":
            depth = 1
            j = pos
            while depth and j < n:  # nested dicts
                a, b = data.find(b"<<", j), data.find(b">>", j)
                if b == -1:
                    raise Unsupported("unterminated dict")
                if a != -1 and a < b:
                    depth += 1
                    j = a + 2
                else:
                    depth -= 1
                    j = b + 2
            stack[-1].append(Raw(data[m.start("dict"):j]))
            pos = j
        elif kind == "hex":
            stack[-1].append(Raw(m.group("hex")))
        elif kind == "str":
            end = _scan_string(data, pos)
            stack[-1].append(Raw(data[m.start("str"):end]))
            pos = end
        # comments ignored
    if len(stack) != 1:
        raise Unsupported("unbalanced array")
    return ops


def _fmt(v, nd: int) -> bytes:
    if isinstance(v, int):
        return str(v).encode()
    s = f"{v:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s.encode()


def _ser_operand(v, nd: int) -> bytes:
    if isinstance(v, (Name, Raw)):
        return bytes(v)
    if isinstance(v, list):
        return b"[" + b" ".join(_ser_operand(x, nd) for x in v) + b"]"
    return _fmt(v, nd)


_MATRIX_OPS = {b"cm", b"Tm"}


def serialize(ops, nd: int) -> bytes:
    out = []
    for op, args in ops:
        if args:
            p = max(nd, 5) if op in _MATRIX_OPS else nd
            out.append(b" ".join(_ser_operand(a, p) for a in args) + b" " + op)
        else:
            out.append(op)
    return b"\n".join(out) + b"\n"


# ----------------------------------------------------------------------------
# Generic fallback: fewer decimals
# ----------------------------------------------------------------------------

def _page_streams(doc, page) -> list[int]:
    return list(page.get_contents())


def round_content_numbers(doc: pymupdf.Document, nd: int, min_size: int = 20_000, skip: set | None = None) -> int:
    """Re-serialise large page content streams with nd decimals. Returns bytes saved (decoded)."""
    saved = 0
    for page in doc:
        if skip and page.number in skip:
            continue
        xrefs = _page_streams(doc, page)
        if not xrefs:
            continue
        data = b"\n".join(doc.xref_stream(x) or b"" for x in xrefs)
        if len(data) < min_size:
            continue
        try:
            ops = parse_ops(data)
        except Unsupported:
            continue
        new = serialize(ops, nd)
        if len(new) >= len(data):
            continue
        _set_contents(doc, page, new)
        saved += len(data) - len(new)
    return saved


def _set_contents(doc, page, data: bytes) -> None:
    xrefs = _page_streams(doc, page)
    first = xrefs[0]
    doc.update_stream(first, zlib.compress(data, 9), compress=0)
    doc.xref_set_key(first, "Filter", "/FlateDecode")
    doc.xref_set_key(first, "DecodeParms", "null")
    if len(xrefs) > 1:
        page.set_contents(first)


# ----------------------------------------------------------------------------
# Outline-text transplant
# ----------------------------------------------------------------------------

IDENT = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
_PATH_OPS = {b"m", b"l", b"c", b"v", b"y", b"h", b"re"}
_FILL_OPS = {b"f", b"F", b"f*"}
_PAINT_OPS = _FILL_OPS | {b"S", b"s", b"B", b"B*", b"b", b"b*", b"n"}
_FILL_COLOR_OPS = {b"g", b"rg", b"k", b"cs", b"sc", b"scn"}


def _mmul(m, n):
    return (
        m[0] * n[0] + m[1] * n[2], m[0] * n[1] + m[1] * n[3],
        m[2] * n[0] + m[3] * n[2], m[2] * n[1] + m[3] * n[3],
        m[4] * n[0] + m[5] * n[2] + n[4], m[4] * n[1] + m[5] * n[3] + n[5],
    )


def _minv(m):
    det = m[0] * m[3] - m[1] * m[2]
    if abs(det) < 1e-12:
        raise Unsupported("singular matrix")
    a, b, c, d = m[3] / det, -m[1] / det, -m[2] / det, m[0] / det
    return (a, b, c, d, -(m[4] * a + m[5] * c), -(m[4] * b + m[5] * d))


def _apply(m, x, y):
    return m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]


class _SubPath:
    __slots__ = ("i0", "i1", "pts", "bbox", "inst", "ok", "degenerate", "fill")

    def __init__(self, i0):
        self.i0, self.i1, self.pts, self.bbox, self.inst, self.ok = i0, i0, [], None, None, True
        self.degenerate, self.fill = False, None


class _Fill:
    __slots__ = ("subs", "paint_idx", "rule", "color", "ctm", "clip")


class _Inst:
    __slots__ = ("font", "code", "M", "bbox", "show", "subs", "valid", "group", "k", "cw")


class _Show:
    __slots__ = ("op_idx", "insts", "bt_idx", "state", "clip", "pen0")


class _Font:
    def __init__(self, xref, procs, enc, widths, first, bbox_of, ws):
        self.xref, self.procs, self.enc, self.widths, self.first = xref, procs, enc, widths, first
        self.bbox_of = bbox_of  # code -> d1 operands (wx, wy, llx, lly, urx, ury)
        self.ws = ws  # codes that ToUnicode maps to whitespace: they can never have ink


def _whitespace_codes(doc, font_xref) -> set:
    tu = doc.xref_get_key(font_xref, "ToUnicode")
    if tu[0] != "xref":
        return set()
    text = (doc.xref_stream(int(tu[1].split()[0])) or b"").decode("latin1")
    ws = set()

    def uni(h):
        try:
            return bytes.fromhex(h).decode("utf-16-be")
        except Exception:
            return "?"
    for block in re.findall(r"beginbfchar(.*?)endbfchar", text, re.S):
        for src, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            u = uni(dst)
            if u and (u.isspace() or set(u) <= {"\u200b", "\u200c", "\u200d", "\ufeff"}):
                ws.add(int(src, 16))
    for block in re.findall(r"beginbfrange(.*?)endbfrange", text, re.S):
        for lo, hi, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            u = uni(dst)
            if u and u.isspace():
                ws.update(range(int(lo, 16), int(hi, 16) + 1))
    return ws


def _obj_text(doc, xref) -> str:
    return doc.xref_object(xref, compressed=True)


def _resolve(doc, text: str) -> str:
    """If text is 'N 0 R', fetch that object's text."""
    m = re.fullmatch(r"\s*(\d+) 0 R\s*", text)
    return _obj_text(doc, int(m.group(1))) if m else text


_EMPTY_D1 = re.compile(rb"\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+d1\s*")
_EMPTY_D0 = re.compile(rb"\s*(-?[\d.]+)\s+(-?[\d.]+)\s+d0\s*")


def _load_type3(doc, xref) -> _Font | None:
    """A Type 3 font whose glyphs are all empty (the invisible text layer), else None."""
    try:
        if doc.xref_get_key(xref, "Subtype")[1] != "/Type3":
            return None
        matrix = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+", doc.xref_get_key(xref, "FontMatrix")[1])]
        if matrix != [1, 0, 0, 1, 0, 0]:
            return None
        cp = doc.xref_get_key(xref, "CharProcs")[1]
        procs = {m.group(1): int(m.group(2)) for m in re.finditer(r"/([^\s/<>\[\]]+)\s+(\d+) 0 R", cp)}
        enc_text = _resolve(doc, doc.xref_get_key(xref, "Encoding")[1])
        diff = re.search(r"/Differences\s*\[(.*?)\]", enc_text, re.S)
        if not diff:
            return None
        enc, code = {}, 0
        for tok in re.findall(r"/[^\s/\[\]]+|\d+", diff.group(1)):
            if tok.startswith("/"):
                enc[code] = tok[1:]
                code += 1
            else:
                code = int(tok)
        first = int(doc.xref_get_key(xref, "FirstChar")[1])
        widths = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+", _resolve(doc, doc.xref_get_key(xref, "Widths")[1]))]
        bbox_of = {}
        for c, name in enc.items():
            px = procs.get(name)
            if px is None:
                continue
            stream = doc.xref_stream(px) or b""
            m = _EMPTY_D1.fullmatch(stream)
            if m:
                bbox_of[c] = tuple(float(v) for v in m.groups())
            elif not _EMPTY_D0.fullmatch(stream):
                return None  # this glyph draws something: a real Type 3 font, not our business
        return _Font(xref, procs, enc, widths, first, bbox_of, _whitespace_codes(doc, xref))
    except Exception:
        return None


def _page_fonts(doc, page) -> dict[bytes, _Font]:
    fonts = {}
    for f in page.get_fonts(full=True):
        xref, name = f[0], f[4]
        t3 = _load_type3(doc, xref)
        if t3:
            fonts[b"/" + name.encode()] = t3
    return fonts


class _PageScan:
    """Walks a page content stream and records outline fills, glyph instances, text shows."""

    def __init__(self, doc, page):
        self.doc, self.page, self.page_no = doc, page, page.number
        data = b"\n".join(doc.xref_stream(x) or b"" for x in page.get_contents())
        self.ops = parse_ops(data)
        self.fonts = _page_fonts(doc, page)
        self.fills: list[_Fill] = []
        self.insts: list[_Inst] = []
        self.shows: list[_Show] = []
        self.bt_ranges: list[tuple[int, int]] = []
        self._walk()

    def _walk(self):
        ctm, stack = IDENT, []
        fill_state = {"cs": None, "sc": None, "gs": None}
        clip_sig, clip_n = (), 0
        tm = tlm = IDENT
        font = None
        fsize, tc, tw, th, tl, rise = 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
        cur_subs: list[_SubPath] = []
        cur = None
        clip_pending = False
        bt_start = None
        for i, (op, a) in enumerate(self.ops):
            if op == b"q":
                stack.append((ctm, dict(fill_state), font, fsize, tc, tw, th, tl, rise, clip_sig))
            elif op == b"Q":
                if stack:
                    ctm, fill_state, font, fsize, tc, tw, th, tl, rise, clip_sig = stack.pop()
            elif op == b"cm":
                ctm = _mmul(tuple(float(v) for v in a[:6]), ctm)
            elif op in _FILL_COLOR_OPS:
                gs = fill_state["gs"]
                if op == b"cs":
                    fill_state = {"cs": (op, a), "sc": None, "gs": gs}
                elif op in (b"sc", b"scn"):
                    fill_state["sc"] = (op, a)
                else:
                    fill_state = {"cs": None, "sc": (op, a), "gs": gs}
            elif op == b"gs":
                fill_state["gs"] = (op, a)
            elif op in _PATH_OPS:
                if op == b"m":
                    cur = _SubPath(i)
                    cur_subs.append(cur)
                elif cur is None:
                    raise Unsupported("path without moveto")
                if op == b"re":
                    cur = _SubPath(i)
                    cur.ok = False
                    cur_subs.append(cur)
                if op != b"h":
                    nums = [float(v) for v in a]
                    for k in range(0, len(nums), 2):
                        cur.pts.append(_apply(ctm, nums[k], nums[k + 1]))
                    if op == b"re":
                        x, y, w, h = nums
                        cur.pts = [_apply(ctm, x, y), _apply(ctm, x + w, y), _apply(ctm, x, y + h), _apply(ctm, x + w, y + h)]
                cur.i1 = i + 1
            elif op in (b"W", b"W*"):
                clip_pending = True
                clip_n += 1
                clip_sig = clip_sig + (clip_n,)
            elif op in _PAINT_OPS:
                if op in _FILL_OPS and cur_subs and not clip_pending:
                    f = _Fill()
                    f.subs, f.paint_idx, f.rule = cur_subs, i, op
                    f.color = (fill_state["cs"], fill_state["sc"], fill_state["gs"])
                    f.ctm, f.clip = ctm, clip_sig
                    for s in cur_subs:
                        xs = [p[0] for p in s.pts]
                        ys = [p[1] for p in s.pts]
                        s.bbox = (min(xs), min(ys), max(xs), max(ys)) if xs else None
                        s.fill = f
                        s.degenerate = bool(xs) and len(xs) == 1 and s.ok
                    self.fills.append(f)
                cur_subs, cur, clip_pending = [], None, False
            elif op == b"BT":
                tm = tlm = IDENT
                bt_start = i
            elif op == b"ET":
                if bt_start is not None:
                    self.bt_ranges.append((bt_start, i))
                bt_start = None
            elif op == b"Tf":
                font = self.fonts.get(bytes(a[0])) or ("other", bytes(a[0]))
                fsize = float(a[1])
            elif op == b"Tc":
                tc = float(a[0])
            elif op == b"Tw":
                tw = float(a[0])
            elif op == b"Tz":
                th = float(a[0]) / 100
            elif op == b"TL":
                tl = float(a[0])
            elif op == b"Ts":
                rise = float(a[0])
            elif op == b"Tr":
                raise Unsupported("text render mode in use")
            elif op == b"Tm":
                tm = tlm = tuple(float(v) for v in a[:6])
            elif op in (b"Td", b"TD"):
                if op == b"TD":
                    tl = -float(a[1])
                tlm = _mmul((1, 0, 0, 1, float(a[0]), float(a[1])), tlm)
                tm = tlm
            elif op == b"T*":
                tlm = _mmul((1, 0, 0, 1, 0, -tl), tlm)
                tm = tlm
            elif op in (b"Tj", b"TJ", b"'", b'"'):
                if op in (b"'", b'"'):
                    raise Unsupported("quote text operators")
                if isinstance(font, tuple):
                    raise Unsupported("text in a non-Type3 font on a page we rewrite")
                if font is None:
                    raise Unsupported("text without font")
                show = _Show()
                show.op_idx, show.insts, show.bt_idx = i, [], bt_start
                show.state = (fill_state["cs"], fill_state["sc"], fill_state["gs"])
                show.clip = clip_sig
                show.pen0 = _mmul(_mmul((fsize * th, 0, 0, fsize, 0, rise), tm), ctm)[4]
                items = a[0] if op == b"TJ" else [a[0]]
                for it in items:
                    if isinstance(it, (int, float)):
                        tm = _mmul((1, 0, 0, 1, -float(it) / 1000 * fsize * th, 0), tm)
                        continue
                    for code in _string_codes(bytes(it)):
                        w0 = 0.0
                        wi = code - font.first
                        if 0 <= wi < len(font.widths):
                            w0 = font.widths[wi]
                        bb = font.bbox_of.get(code)
                        inst = _Inst()
                        inst.font, inst.code, inst.show, inst.subs, inst.valid = font, code, show, [], False
                        inst.group = None
                        # glyph space -> page space (font matrix is identity here)
                        M = _mmul(_mmul((fsize * th, 0, 0, fsize, 0, rise), tm), ctm)
                        inst.M = M
                        inst.k = M[0] / (fsize * th) if fsize * th else 1.0
                        inst.cw = (tc + (tw if code == 32 else 0)) * th
                        if bb and bb[4] > bb[2] and bb[5] > bb[3]:
                            llx, lly, urx, ury = bb[2:6]
                            cs = [_apply(M, x, y) for x, y in ((llx, lly), (urx, lly), (llx, ury), (urx, ury))]
                            inst.bbox = (min(c[0] for c in cs), min(c[1] for c in cs), max(c[0] for c in cs), max(c[1] for c in cs))
                        else:
                            inst.bbox = None
                        self.insts.append(inst)
                        show.insts.append(inst)
                        tx = (w0 * fsize + tc + (tw if code == 32 else 0)) * th
                        tm = _mmul((1, 0, 0, 1, tx, 0), tm)
                self.shows.append(show)


def _string_codes(raw: bytes) -> list[int]:
    if raw.startswith(b"<"):
        h = re.sub(rb"\s", b"", raw[1:-1])
        if len(h) % 2:
            h += b"0"
        return list(bytes.fromhex(h.decode()))
    s = raw[1:-1]
    out, i = [], 0
    esc = {ord("n"): 10, ord("r"): 13, ord("t"): 9, ord("b"): 8, ord("f"): 12,
           ord("("): 40, ord(")"): 41, ord("\\"): 92}
    while i < len(s):
        c = s[i]
        if c == 0x5C and i + 1 < len(s):
            d = s[i + 1]
            if 48 <= d <= 55:
                j = i + 1
                while j < len(s) and j < i + 4 and 48 <= s[j] <= 55:
                    j += 1
                out.append(int(s[i + 1:j], 8) & 255)
                i = j
                continue
            if d in (10, 13):
                i += 2
                continue
            out.append(esc.get(d, d))
            i += 2
            continue
        out.append(c)
        i += 1
    return out


class _Group:
    """One glyph's worth of outline: the zero-area marker at its origin plus the subpaths after it."""
    __slots__ = ("origin", "fill", "subs", "inst", "metrics")

    def __init__(self, origin, fill):
        self.origin, self.fill, self.subs, self.inst, self.metrics = origin, fill, [], None, None


MAX_ALIGN = 1.0  # pt: farthest an outline origin may sit from its text-layer glyph origin
OFFSET_RANGE = 12.0  # pt: largest constant shift between a text run and its outlines that is searched
WIDE_ALIGN = 3.0  # pt: residual tolerance after the run's shift has been removed
SKIP_INST, SKIP_MARK = 0.45, 1.0
SHAPE_TOL = 0.012  # em: tolerance when deciding that an outline "is" a known glyph
CONSIST_TOL = 0.006  # em: max deviation between instances of the same glyph
ADJ_MIN = 0.4  # thousandths of an em: smaller position corrections are not worth a number


def _align_run(xi, xm, allowed=None, max_dx=MAX_ALIGN, base=0.0, slope=1.0):
    """Monotone alignment of glyph origins to outline markers (both sorted by x).
    A pair costs base + slope*|dx|; skipping a glyph / a marker has a fixed cost."""
    n, m = len(xi), len(xm)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    B = [[0] * (m + 1) for _ in range(n + 1)]  # 1 match, 2 skip glyph, 3 skip marker
    for i in range(1, n + 1):
        D[i][0], B[i][0] = i * SKIP_INST, 2
    for j in range(1, m + 1):
        D[0][j], B[0][j] = j * SKIP_MARK, 3
    for i in range(1, n + 1):
        xv, Di, Dp = xi[i - 1], D[i], D[i - 1]
        for j in range(1, m + 1):
            dx = abs(xv - xm[j - 1])
            best, how = Dp[j] + SKIP_INST, 2
            c = Di[j - 1] + SKIP_MARK
            if c < best:
                best, how = c, 3
            if dx <= max_dx:
                c = Dp[j - 1] + base + slope * dx
                if c < best and (allowed is None or allowed(i - 1, j - 1)):
                    best, how = c, 1
            Di[j], B[i][j] = best, how
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        how = B[i][j]
        if how == 1:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif how == 2:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _estimate_offset(xi, xm) -> float:
    """Shift t (pt) that best lays the text-layer glyph origins xi onto the outline origins xm."""
    import bisect
    if len(xi) < 6 or not xm:
        return 0.0
    best_t, best_score = 0.0, -1.0
    steps = int(2 * OFFSET_RANGE / 0.1)
    for k in range(steps + 1):
        t = -OFFSET_RANGE + 0.1 * k
        score = 0
        for x in xi:
            j = bisect.bisect_left(xm, x + t - 1.0)
            if j < len(xm) and xm[j] <= x + t + 1.0:
                score += 1
        score -= 0.002 * abs(t)  # prefer the smaller shift on ties
        if score > best_score:
            best_t, best_score = t, score
    # refine on the matched pairs
    diffs = []
    for x in xi:
        j = bisect.bisect_left(xm, x + best_t - 1.0)
        if j < len(xm) and xm[j] <= x + best_t + 1.0:
            diffs.append(xm[j] - x)
    diffs.sort()
    return diffs[len(diffs) // 2] if diffs else best_t


_RECT_SIGS = {(b"m", b"l", b"l", b"l", b"h"), (b"m", b"l", b"l", b"l", b"l", b"h")}


def _rule_like(scan: _PageScan, sp: _SubPath) -> bool:
    """A thin, wide rectangle: part of an underline or rule rather than a glyph."""
    if not sp.bbox:
        return False
    w, h = sp.bbox[2] - sp.bbox[0], sp.bbox[3] - sp.bbox[1]
    if not (h <= 1.5 and w >= 3 * max(h, 0.05)):
        return False
    return tuple(op for op, _ in scan.ops[sp.i0:sp.i1]) in _RECT_SIGS


def _collect_groups(scan: _PageScan) -> dict:
    groups: list[_Group] = []
    by_y: dict[int, list[_Group]] = defaultdict(list)
    for fill in scan.fills:
        cur = None
        for sp in fill.subs:
            if not sp.ok or not sp.bbox:
                cur = None
                continue
            if sp.degenerate:
                cur = _Group(sp.pts[0], fill)
                groups.append(cur)
                by_y[round(cur.origin[1] * 10)].append(cur)
            elif cur is not None:
                cur.subs.append(sp)
    for g in groups:
        while len(g.subs) > 1 and _rule_like(scan, g.subs[-1]):
            g.subs.pop()  # underline / strike segments belong to the page, not to the glyph
        if g.subs:
            pts = [p for sp in g.subs for p in sp.pts]
            x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
            y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
            sig = tuple(tuple(op for op, _ in scan.ops[sp.i0:sp.i1]) for sp in g.subs)
            g.metrics = (sig, x0 - g.origin[0], y0 - g.origin[1], x1 - x0, y1 - y0)
    for lst in by_y.values():
        lst.sort(key=lambda g: g.origin[0])
    scan.groups, scan.by_y = groups, by_y
    return by_y


def _reset_matches(scan):
    for i in scan.insts:
        i.group, i.valid, i.subs = None, False, []
    for g in scan.groups:
        g.inst = None


def _inside_expected(g: _Group, inst: _Inst) -> bool:
    """Every subpath of the group must lie within the glyph's declared box (plus slack)."""
    if not inst.bbox:
        return True
    em = abs(inst.M[0]) or 1.0
    ox, oy = g.origin
    bx0, by0, bx1, by1 = inst.bbox
    # the declared box is positioned by the text layer; compare in the outline's own frame
    dx, dy = ox - inst.M[4], oy - inst.M[5]
    bx0, bx1, by0, by1 = bx0 + dx, bx1 + dx, by0 + dy, by1 + dy
    t = 0.08 * em + 0.05
    for sp in g.subs:
        x0, y0, x1, y1 = sp.bbox
        if x0 < bx0 - t or x1 > bx1 + t or y0 < by0 - t or y1 > by1 + 0.25 * em + t:
            return False
    return True


def _match(scan: _PageScan, canon: dict | None) -> dict:
    """Attach each glyph's outline group to the text-layer glyph it belongs to.
    canon (font, code) -> metrics in em, when known, restricts pairs to matching shapes."""
    _reset_matches(scan)
    drifts = []
    for show in scan.shows:
        if not show.insts:
            continue
        for inst in show.insts:
            if abs(inst.M[1]) > 1e-6 or abs(inst.M[2]) > 1e-6:
                raise Unsupported("rotated or skewed text")
        ink = [i for i in show.insts if i.code not in i.font.ws]
        if not ink:
            continue
        y0 = ink[0].M[5]
        xi = [i.M[4] for i in ink]
        lo, hi = min(xi) - OFFSET_RANGE - 2, max(xi) + OFFSET_RANGE + 2
        cands = []
        for ky in (round(y0 * 10) - 1, round(y0 * 10), round(y0 * 10) + 1):
            cands += [g for g in scan.by_y.get(ky, ()) if g.subs and abs(g.origin[1] - y0) < 0.08 and lo <= g.origin[0] <= hi]
        if not cands:
            continue
        cands.sort(key=lambda g: g.origin[0])
        shift = _estimate_offset(xi, [g.origin[0] for g in cands])
        xi = [x + shift for x in xi]

        allowed = None
        if canon is not None:
            def allowed(i, j, ink=ink, cands=cands):
                inst = ink[i]
                c = canon.get((inst.font.xref, inst.code))
                if c is None:
                    return False
                sig, cx0, cy0, cw, ch = c
                gsig, gx0, gy0, gw, gh = cands[j].metrics
                if gsig != sig:
                    return False
                em = abs(inst.M[0]) or 1.0
                t = SHAPE_TOL * em + 0.02
                return (abs(gx0 - cx0 * em) <= t and abs(gy0 - cy0 * em) <= t
                        and abs(gw - cw * em) <= t and abs(gh - ch * em) <= t)

        run_kw = {"max_dx": 1.6} if canon is None else {"max_dx": WIDE_ALIGN, "base": 0.1, "slope": 0.05}
        for ii, jj in _align_run(xi, [g.origin[0] for g in cands], allowed, **run_kw):
            g, inst = cands[jj], ink[ii]
            if g.inst is None:
                if g.fill.clip != inst.show.clip or not _inside_expected(g, inst):
                    continue
                g.inst, inst.group, inst.subs, inst.valid = inst, g, g.subs, True
                g.inst = inst
                drifts.append(abs(inst.M[4] + shift - g.origin[0]))
    matched = sum(1 for i in scan.insts if i.valid)
    return {"instances": len(scan.insts), "groups": sum(1 for g in scan.groups if g.subs), "matched": matched,
            "max_drift": round(max(drifts, default=0), 3), "mean_drift": round(sum(drifts) / max(1, len(drifts)), 3)}


MIN_MATCH_FRAC = 0.2  # codes that match fewer of their occurrences than this are treated as inkless


def _drop_weak(scans, min_frac=MIN_MATCH_FRAC) -> set:
    """Un-match every instance of a code whose outlines were found for too few occurrences."""
    tot, hit = defaultdict(int), defaultdict(int)
    for scan in scans:
        for inst in scan.insts:
            k = (inst.font.xref, inst.code)
            tot[k] += 1
            hit[k] += inst.group is not None
    weak = {k for k in tot if hit[k] / tot[k] < min_frac}
    for scan in scans:
        for inst in scan.insts:
            if (inst.font.xref, inst.code) in weak and inst.group is not None:
                inst.group.inst = None
                inst.group, inst.valid, inst.subs = None, False, []
    return weak


def _canon_from_matches(scans) -> dict:
    """Majority-vote glyph metrics (in em) per (font, code) from the current matches."""
    _drop_weak(scans)
    votes = defaultdict(lambda: defaultdict(list))
    for scan in scans:
        for inst in scan.insts:
            if inst.group is None:
                continue
            sig, x0, y0, w, h = inst.group.metrics
            em = abs(inst.M[0]) or 1.0
            key = (sig, round(x0 / em / 0.02), round(y0 / em / 0.02), round(w / em / 0.02), round(h / em / 0.02))
            votes[(inst.font.xref, inst.code)][key].append((x0 / em, y0 / em, w / em, h / em))
    canon = {}
    for fc, d in votes.items():
        key, members = max(d.items(), key=lambda kv: len(kv[1]))
        n = len(members)
        canon[fc] = (key[0], *(sum(m[i] for m in members) / n for i in range(4)))
    return canon


def _pair_shape(scan: _PageScan, inst: _Inst):
    """Outline of a matched glyph in glyph space (em = 1)."""
    g = inst.group
    M = (inst.M[0], inst.M[1], inst.M[2], inst.M[3], g.origin[0], g.origin[1])
    T = _mmul(g.fill.ctm, _minv(M))
    shape = []
    for sp in g.subs:
        cur = None
        for op, a in scan.ops[sp.i0:sp.i1]:
            nums = [float(v) for v in a]
            pts = []
            for k in range(0, len(nums), 2):
                pts += list(_apply(T, nums[k], nums[k + 1]))
            if op == b"v":
                pts, op = list(cur) + pts, b"c"
            elif op == b"y":
                pts, op = pts[:2] + pts[2:4] + pts[2:4], b"c"
            if pts:
                cur = pts[-2:]
            shape.append((op, pts))
    return shape


def _shape_close_tol(a, b, tol) -> bool:
    if len(a) != len(b):
        return False
    for (o1, p1), (o2, p2) in zip(a, b):
        if o1 != o2 or len(p1) != len(p2) or any(abs(x - y) > tol for x, y in zip(p1, p2)):
            return False
    return True


def _shape_close(a, b) -> bool:
    return _shape_close_tol(a, b, CONSIST_TOL)


def _final_shapes(scans) -> tuple[dict, int]:
    """Canonical outline per (font, code) = the most common shape; disagreeing instances are
    un-matched. -> ({(font, code): (shape, fill rule)}, number of outliers)"""
    per = defaultdict(list)
    for scan in scans:
        for inst in scan.insts:
            if inst.group is not None:
                per[(inst.font.xref, inst.code)].append((_pair_shape(scan, inst), inst))
    ref, outliers = {}, 0
    for key, lst in per.items():
        bins = defaultdict(list)
        for shape, inst in lst:
            sig = tuple((op, tuple(round(v * 100) for v in pts)) for op, pts in shape)
            bins[(sig, inst.group.fill.rule)].append((shape, inst))
        best = max(bins.values(), key=len)
        rshape, rrule = best[0][0], best[0][1].group.fill.rule
        xs = [v for _, pts in rshape for v in pts[0::2]]
        ys = [v for _, pts in rshape for v in pts[1::2]]
        if min(xs) < -0.5 or max(xs) > 2.0 or min(ys) < -0.8 or max(ys) > 1.6:
            # not a plausible letter (a rule or ornament got attached): keep it as plain vector art
            for shape, inst in lst:
                outliers += 1
                inst.group.inst = None
                inst.group, inst.valid, inst.subs = None, False, []
            continue
        ref[key] = (rshape, rrule)
        for shape, inst in lst:
            if inst.group.fill.rule != rrule or not _shape_close(shape, rshape):
                outliers += 1
                inst.group.inst = None
                inst.group, inst.valid, inst.subs = None, False, []
    weak = _drop_weak(scans)
    for k in weak:
        ref.pop(k, None)
    return ref, outliers


def _estimate_widths(scans, shapes, orig_widths) -> dict:
    """(font, code) -> advance in em, from spacing between consecutive matched glyphs."""
    obs = defaultdict(list)
    for scan in scans:
        for show in scan.shows:
            prev = None
            for inst in show.insts:
                if inst.group is not None:
                    if prev is not None and prev.group is not None and prev.font is inst.font:
                        em = abs(prev.M[0]) or 1.0
                        obs[(prev.font.xref, prev.code)].append((inst.group.origin[0] - prev.group.origin[0]) / em)
                prev = inst
    widths = {}
    for scan in scans:
        for inst in scan.insts:
            key = (inst.font.xref, inst.code)
            if key in widths:
                continue
            vals = sorted(obs.get(key, ()))
            if vals and key in shapes:
                widths[key] = round(vals[len(vals) // 2], 4)
            else:
                widths[key] = orig_widths[key]
    return widths


# ---------------------------------------------------------------- CFF font building

def _build_cff(psname: str, glyphs: dict[str, list]) -> bytes:
    """glyphs: glyph name -> shape (em units). Returns a bare CFF table (for a PDF Type1C font)."""
    import io
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.t2CharStringPen import T2CharStringPen
    from fontTools.ttLib import TTFont

    order = [".notdef"] + list(glyphs)
    fb = FontBuilder(1000, isTTF=False)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap({})
    cs = {}
    for name in order:
        pen = T2CharStringPen(None, None)
        for op, pts in glyphs.get(name, ()):
            v = [int(round(x * 1000)) for x in pts]
            if op == b"m":
                pen.moveTo((v[0], v[1]))
            elif op == b"l":
                pen.lineTo((v[0], v[1]))
            elif op == b"c":
                pen.curveTo((v[0], v[1]), (v[2], v[3]), (v[4], v[5]))
            elif op == b"h":
                pen.closePath()
            else:
                raise Unsupported(f"path operator {op!r} in glyph")
        cs[name] = pen.getCharString()
    fb.setupCFF(psname, {"FullName": psname, "FamilyName": psname, "Weight": "Regular"}, cs, {})
    fb.setupHorizontalMetrics({n: (0, 0) for n in order})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": psname, "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=800, usWinAscent=800, usWinDescent=200)
    fb.setupPost()
    try:  # shared subroutines shrink the charstrings noticeably; optional
        import compreffor
        compreffor.compress(fb.font)
    except Exception:
        pass
    buf = io.BytesIO()
    fb.font.save(buf)
    buf.seek(0)
    return TTFont(buf).reader["CFF "]


# ---------------------------------------------------------------- rewriting

def _pdf_string(codes) -> bytes:
    out = bytearray(b"(")
    for c in codes:
        if c in (40, 41, 92):
            out += b"\\" + bytes([c])
        elif 32 <= c < 127:
            out.append(c)
        else:
            out += b"\\%03o" % c
    out += b")"
    return bytes(out)


def _color_ops(color):
    ops = []
    for part in color:  # colour space, colour, graphics-state dictionary (transparency etc.)
        if part:
            ops.append((part[0], list(part[1])))
    return ops


def _rewrite_show(show, widths_em, shapes, convert):
    """New operators for one text-showing op.

    convert=True : matched glyphs are drawn by the shared font exactly where their outline was (colour taken
                   from the outline); glyphs that have ink but no matching outline are hidden.
    convert=False: the text stays invisible (its outlines remain as ordinary paths), but it is still
                   re-spaced onto those outlines so selection and extraction line up with what you see.
    """
    segs = []  # [hidden | None, colour | None, items (Raw | float), codes]

    def new_seg(hidden, color):
        segs.append([hidden, color, [], []])

    def flush_codes(seg):
        if seg[3]:
            seg[2].append(Raw(_pdf_string(seg[3])))
            seg[3] = []

    pen = show.pen0
    new_seg(None, None)
    for inst in show.insts:
        key = (inst.font.xref, inst.code)
        em = abs(inst.M[0]) or 1.0
        g = inst.group
        if convert:
            hide = False if g is not None else (True if key in shapes else None)
            color = g.fill.color if g is not None else None
        else:
            hide = True if inst.code not in inst.font.ws else None
            color = None
        cur = segs[-1]
        split = ((hide is not None and cur[0] is not None and hide != cur[0]) or
                 (color is not None and cur[1] is not None and repr(color) != repr(cur[1])))
        if split:
            flush_codes(cur)
            new_seg(hide, color)
            cur = segs[-1]
        else:
            if hide is not None and cur[0] is None:
                cur[0] = hide
            if color is not None and cur[1] is None:
                cur[1] = color
        if g is not None:
            n = round(-(g.origin[0] - pen) / em * 1000, 1)
            if abs(n) >= ADJ_MIN:
                flush_codes(cur)
                cur[2].append(n)
                pen += -n / 1000 * em
        cur[3].append(inst.code)
        pen += widths_em[key] * em + inst.cw * inst.k
    flush_codes(segs[-1])

    ops, cur_color, changed_color = [], None, False
    for hidden, color, items, _ in segs:
        if not items:
            continue
        if hidden:
            ops += [(b"Tr", [3]), (b"TJ", [items]), (b"Tr", [0])]
        else:
            if color is not None and repr(color) != repr(cur_color if cur_color is not None else show.state):
                ops += _color_ops(color)
                cur_color = color
                changed_color = True
            ops.append((b"TJ", [items]))
    return ops, changed_color


def _slim_tounicode(doc, font_xref, used) -> int:
    """Rewrite a ToUnicode CMap to list only the codes actually used. Returns the new raw size."""
    tu = doc.xref_get_key(font_xref, "ToUnicode")
    if tu[0] != "xref":
        return 0
    ux = int(tu[1].split()[0])
    text = (doc.xref_stream(ux) or b"").decode("latin1")
    m = {}
    for blk in re.findall(r"beginbfchar(.*?)endbfchar", text, re.S):
        for src, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", blk):
            m[int(src, 16)] = dst
    for blk in re.findall(r"beginbfrange(.*?)endbfrange", text, re.S):
        for lo, hi, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", blk):
            base = int(dst, 16)
            for k, c in enumerate(range(int(lo, 16), int(hi, 16) + 1)):
                m[c] = f"{base + k:0{len(dst)}X}"
    if not m:
        return 0
    rows = [f"<{c:02X}> <{m[c]}>" for c in sorted(used) if c in m]
    body = []
    for i in range(0, len(rows), 100):
        chunk = rows[i:i + 100]
        body.append(f"{len(chunk)} beginbfchar\n" + "\n".join(chunk) + "\nendbfchar")
    cmap = ("/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
            "/CIDSystemInfo<</Registry(Adobe)/Ordering(UCS)/Supplement 0>>def\n/CMapName/Adobe-Identity-UCS def\n"
            "/CMapType 2 def\n1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
            + "\n".join(body) + "\nendcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n").encode()
    old = len(doc.xref_stream_raw(ux))
    z = zlib.compress(cmap, 9)
    if len(z) < old:
        doc.update_stream(ux, z, compress=0)
        doc.xref_set_key(ux, "Filter", "/FlateDecode")
    return len(cmap)


def _build_glyph_classes(shapes):
    """Identical outlines (within tolerance) are stored once. -> (cls_of, glyphs name->shape)"""
    classes: list[tuple[str, list]] = []
    buckets = defaultdict(list)
    cls_of = {}
    for key in sorted(shapes):
        shape, rule = shapes[key]
        xs = [v for _, pts in shape for v in pts[0::2]]
        ys = [v for _, pts in shape for v in pts[1::2]]
        bk = (tuple(op for op, _ in shape), rule, round(min(xs) / 0.02), round(max(xs) / 0.02),
              round(min(ys) / 0.02), round(max(ys) / 0.02))
        found = None
        for cand in buckets[bk]:
            if _shape_close_tol(shape, classes[cand][1], 0.0015):
                found = cand
                break
        if found is None:
            found = len(classes)
            classes.append((f"s{found}", shape))
            buckets[bk].append(found)
        cls_of[key] = found
    return cls_of, {name: shape for name, shape in classes}


class OutlineAnalysis:
    """Everything learned about a document whose text is drawn as outlines. Built once, then applied (fully
    or partly) to fresh copies of the document."""

    def __init__(self, scans, shapes, widths_em):
        self.scans, self.shapes, self.widths_em = scans, shapes, widths_em
        self.cls_of, self.glyphs = _build_glyph_classes(shapes)
        self.cff = _build_cff("OUTLINE+Shared", self.glyphs)
        xs = [v * 1000 for sh in self.glyphs.values() for _, pts in sh for v in pts[0::2]] or [0]
        ys = [v * 1000 for sh in self.glyphs.values() for _, pts in sh for v in pts[1::2]] or [0]
        self.bbox = (int(min(xs)) - 1, int(min(ys)) - 1, int(max(xs)) + 1, int(max(ys)) + 1)
        # text runs that can be converted, smallest type first (large headings stay exact outlines longest)
        runs = []
        for si, scan in enumerate(scans):
            for show in scan.shows:
                n = sum(1 for i in show.insts if i.group is not None)
                if n:
                    runs.append((abs(show.insts[0].M[0]), si, show.op_idx, show, n))
        runs.sort(key=lambda r: r[:3])
        self.order = [r[3] for r in runs]
        self.cum = [0]
        for r in runs:
            self.cum.append(self.cum[-1] + r[4])


def _install_fonts(doc, an: OutlineAnalysis, shapes_sel):
    """One shared Type1C font program holds every distinct glyph; each Type 3 font dictionary becomes a Type1
    font that maps its own codes onto those glyphs (and keeps its ToUnicode)."""
    used = defaultdict(dict)  # font xref -> {code: width em}
    for scan in an.scans:
        for inst in scan.insts:
            used[inst.font.xref][inst.code] = an.widths_em[(inst.font.xref, inst.code)]

    ff = doc.get_new_xref()
    doc.update_object(ff, "<</Subtype/Type1C>>")
    doc.update_stream(ff, zlib.compress(an.cff, 9), compress=0)
    doc.xref_set_key(ff, "Filter", "/FlateDecode")
    fd = doc.get_new_xref()
    b = an.bbox
    doc.update_object(fd, f"<</Type/FontDescriptor/FontName/OUTLINE+Shared/Flags 4"
                          f"/FontBBox[{b[0]} {b[1]} {b[2]} {b[3]}]"
                          f"/ItalicAngle 0/Ascent 800/Descent -200/CapHeight 700/StemV 80/FontFile3 {ff} 0 R>>")
    for fx, codes in sorted(used.items()):
        thou = {c: round(w * 1000, 1) for c, w in codes.items()}
        first, last = min(codes), max(codes)
        warr = " ".join(f"{thou.get(c, 0):g}" for c in range(first, last + 1))
        diffs, prev = [], None
        for c in sorted(codes):
            if (fx, c) not in shapes_sel:
                continue
            if prev is None or c != prev + 1:
                diffs.append(str(c))
            diffs.append(f"/s{an.cls_of[(fx, c)]}")
            prev = c
        _slim_tounicode(doc, fx, set(codes))
        tu = doc.xref_get_key(fx, "ToUnicode")
        tu_s = f"/ToUnicode {tu[1]}" if tu[0] == "xref" else ""
        doc.update_object(fx, f"<</Type/Font/Subtype/Type1/BaseFont/OUTLINE+Shared/FirstChar {first}/LastChar {last}"
                              f"/Widths[{warr}]/FontDescriptor {fd} 0 R"
                              f"/Encoding<</Type/Encoding/Differences[{' '.join(diffs)}]>>{tu_s}>>")


def _rewrite_page(doc, scan, an, shapes_sel, nd, sel):
    ops = scan.ops
    drop = set()
    for g in scan.groups:
        if g.inst is not None and id(g.inst.show) in sel:
            for sp in g.subs:
                drop.update(range(sp.i0, sp.i1))
    for fill in scan.fills:
        for sp in fill.subs:
            if sp.degenerate:  # zero-area origin markers paint nothing
                drop.update(range(sp.i0, sp.i1))
        if all(sp.i0 in drop for sp in fill.subs):
            drop.add(fill.paint_idx)

    et_of = {a: b for a, b in scan.bt_ranges}
    replace, wrap = {}, {}
    for show in scan.shows:
        new_ops, changed = _rewrite_show(show, an.widths_em, shapes_sel, id(show) in sel)
        replace[show.op_idx] = new_ops
        if changed and show.bt_idx is not None:
            wrap[show.bt_idx] = et_of.get(show.bt_idx)
    out = []
    close_after = set()
    for i, (op, a) in enumerate(ops):
        if i in drop:
            continue
        if i in wrap and wrap[i] is not None:
            out.append((b"q", []))
            close_after.add(wrap[i])
        if i in replace:
            out.extend(replace[i])
        else:
            out.append((op, a))
        if i in close_after:
            out.append((b"Q", []))
    _set_contents(doc, doc[scan.page_no], serialize(out, nd))


def _page_has_text_forms(doc, page) -> bool:
    for x in page.get_xobjects():
        try:
            if re.search(rb"\bTf\b", doc.xref_stream(x[0]) or b""):
                return True
        except Exception:
            return True
    return False


def _check_unrotated(scan: _PageScan) -> None:
    for inst in scan.insts:
        if abs(inst.M[1]) > 1e-6 or abs(inst.M[2]) > 1e-6:
            raise Unsupported("rotated or skewed text")


def analyze_outline_text(doc: pymupdf.Document) -> OutlineAnalysis:
    """Read-only. Raises Unsupported if the document isn't outlined text over an invisible text layer."""
    # every page is judged on its own: pages we don't understand (ordinary fonts, rotated text, text inside
    # forms, ...) are simply left exactly as they are, so a merged file can mix both kinds of page
    scans, skipped = [], []
    for page in doc:
        try:
            if _page_has_text_forms(doc, page):
                raise Unsupported("text inside form XObjects")
            sc = _PageScan(doc, page)
            _check_unrotated(sc)
        except Unsupported:
            skipped.append(page)
            continue
        scans.append(sc)
    if skipped:
        left_alone = {f[0] for page in skipped for f in page.get_fonts(full=True)}
        if left_alone & {inst.font.xref for sc in scans for inst in sc.insts}:
            raise Unsupported("a font is shared with a page that can't be rewritten")
    if not any(sc.shows for sc in scans):
        raise Unsupported("no Type 3 text layer")
    for sc in scans:
        _collect_groups(sc)
    for sc in scans:
        _match(sc, None)
    canon = _canon_from_matches(scans)
    for _ in range(2):
        for sc in scans:
            _match(sc, canon)
        canon = _canon_from_matches(scans)
    shapes, outliers = _final_shapes(scans)
    ink_groups = sum(1 for sc in scans for g in sc.groups if g.subs)
    matched = sum(1 for sc in scans for i in sc.insts if i.group is not None)
    if not shapes or matched < 0.5 * max(1, ink_groups):
        raise Unsupported("outlines do not correspond to the text layer")
    if outliers > 0.01 * matched:
        raise Unsupported("glyph shapes are inconsistent")

    orig = {}
    for sc in scans:
        for i in sc.insts:
            wi = i.code - i.font.first
            orig[(i.font.xref, i.code)] = i.font.widths[wi] if 0 <= wi < len(i.font.widths) else 0.0
    widths_em = {k: round(v * 1000, 1) / 1000 for k, v in _estimate_widths(scans, shapes, orig).items()}
    an = OutlineAnalysis(scans, shapes, widths_em)
    for sc in scans:  # nothing below needs the analysed document itself
        sc.doc = sc.page = None
    return an


def apply_outline_text(doc: pymupdf.Document, an: OutlineAnalysis, runs: int | None = None, nd: int = 3) -> dict:
    """Rewrite doc in place (it must be a fresh copy of the document that was analysed). The first `runs`
    text runs, smallest type first, are redrawn by the shared font; everything else keeps its exact original
    outlines. Callers should work on a throwaway copy: an unexpected error leaves it half rewritten."""
    k = len(an.order) if runs is None else max(0, min(runs, len(an.order)))
    sel = {id(show) for show in an.order[:k]}
    sel_keys = {(i.font.xref, i.code) for sc in an.scans for show in sc.shows if id(show) in sel
                for i in show.insts if i.group is not None}
    shapes_sel = {key: an.shapes[key] for key in sel_keys if key in an.shapes}
    _install_fonts(doc, an, shapes_sel)
    for sc in an.scans:
        _rewrite_page(doc, sc, an, shapes_sel, nd, sel)
    round_content_numbers(doc, nd, skip={sc.page_no for sc in an.scans})  # pages we left alone still get the invisible rounding
    return {"runs": k, "runs_total": len(an.order), "glyphs_replaced": an.cum[k], "glyphs_total": an.cum[-1],
            "distinct_glyphs": len(an.glyphs), "font_bytes": len(an.cff)}


def transplant_outline_text(doc: pymupdf.Document, nd: int = 3) -> dict:
    """Convert all outlined text (analyse + apply)."""
    an = analyze_outline_text(doc)
    info = apply_outline_text(doc, an, None, nd)
    info["outlines_left"] = sum(1 for sc in an.scans for g in sc.groups if g.subs and g.inst is None)
    return info
