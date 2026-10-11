"""Clyro: your documents, simplified. Local web app (compress, merge). Runs only on this device (127.0.0.1)."""
from __future__ import annotations

import io
import json
import os
import re
import threading
import time
import uuid

from flask import Flask, jsonify, request, send_file, send_from_directory

import compressor
import merger

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 600 * 1024 * 1024

RESULTS: dict[str, tuple[bytes, str, float]] = {}  # id -> (pdf bytes, download name, created)
LOCK = threading.Lock()
KEEP_SECONDS, KEEP_MAX = 30 * 60, 16


def _remember(data: bytes, name: str) -> str:
    rid = uuid.uuid4().hex
    now = time.time()
    with LOCK:
        for k in [k for k, v in RESULTS.items() if now - v[2] > KEEP_SECONDS]:
            del RESULTS[k]
        while len(RESULTS) >= KEEP_MAX:
            del RESULTS[min(RESULTS, key=lambda k: RESULTS[k][2])]
        RESULTS[rid] = (data, name, now)
    return rid


def _download_name(filename: str, suffix: str = "-compressed") -> str:
    stem = re.sub(r"\.pdf$", "", os.path.basename(filename or "document"), flags=re.I)
    stem = re.sub(r"[^\w .()-]+", "_", stem).strip() or "document"
    return f"{stem}{suffix}.pdf"


def _looks_like_pdf(data: bytes) -> bool:
    return b"%PDF" in data[:1024]


@app.get("/")
def index():
    resp = send_from_directory(app.static_folder, "index.html")
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.get("/api/ping")
def api_ping():
    """The page asks this on load: if it is missing, the running server is older than the page."""
    return jsonify(ok=True, features=["compress", "merge"])


@app.post("/api/compress")
def api_compress():
    f = request.files.get("file")
    if f is None:
        return jsonify(error="No file received."), 400
    data = f.read()
    if not _looks_like_pdf(data):
        return jsonify(error="That doesn't look like a PDF file."), 400
    try:
        target_kb = min(max(float(request.form.get("target_kb", 100)), 5), 20000)
    except ValueError:
        target_kb = 100
    try:
        out, info = compressor.compress(data, int(target_kb * 1024))
    except compressor.PDFError as e:
        return jsonify(error=str(e)), 422
    except Exception:
        app.logger.exception("compression failed")
        return jsonify(error="Something went wrong while compressing this PDF."), 500
    name = _download_name(f.filename)
    return jsonify(id=_remember(out, name), filename=name, **info)


@app.post("/api/inspect")
def api_inspect():
    """Page count + thumbnail for a file the user added to the merge list."""
    f = request.files.get("file")
    if f is None:
        return jsonify(error="No file received."), 400
    data = f.read()
    if not _looks_like_pdf(data):
        return jsonify(error="That doesn't look like a PDF file."), 422
    try:
        return jsonify(merger.inspect(data, f.filename or "This file"))
    except merger.MergeError as e:
        return jsonify(error=str(e)), 422
    except Exception:
        app.logger.exception("inspect failed")
        return jsonify(error="Couldn't read this PDF."), 500


@app.post("/api/merge")
def api_merge():
    files = request.files.getlist("files")
    try:
        specs = json.loads(request.form.get("pages", "[]"))
    except ValueError:
        specs = []
    if len(files) < 2:
        return jsonify(error="Add at least two PDFs to merge."), 400
    if len(specs) != len(files):
        specs = (specs + [""] * len(files))[:len(files)]
    items = []
    for f, spec in zip(files, specs):
        data = f.read()
        if not _looks_like_pdf(data):
            return jsonify(error=f"{f.filename or 'A file'} doesn't look like a PDF."), 422
        items.append((f.filename or "document.pdf", data, str(spec)))
    stem = re.sub(r"\.pdf$", "", request.form.get("name", "").strip(), flags=re.I)
    name = _download_name(stem or "merged", suffix="")
    try:
        out, info = merger.merge(items, bookmarks=request.form.get("bookmarks", "1") != "0",
                                 title=stem or "Merged")
    except merger.MergeError as e:
        return jsonify(error=str(e)), 422
    except Exception:
        app.logger.exception("merge failed")
        return jsonify(error="Something went wrong while merging these PDFs."), 500
    return jsonify(id=_remember(out, name), filename=name, **info)


@app.get("/api/pdf/<rid>")
def api_pdf(rid: str):
    with LOCK:
        item = RESULTS.get(rid)
    if item is None:
        return jsonify(error="That result has expired. Run it again."), 404
    data, name, _ = item
    return send_file(io.BytesIO(data), mimetype="application/pdf",
                     as_attachment=request.args.get("download") == "1", download_name=name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    reload = os.environ.get("RELOAD", "1") == "1"  # restart by itself when the code changes
    if not reload or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print(f"\n  Clyro is running at http://127.0.0.1:{port}\n  Your documents. Simplified. (files stay on this device)\n")
    app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=reload)
