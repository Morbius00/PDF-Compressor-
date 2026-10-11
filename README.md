# Clyro

**Your documents. Simplified.**  Merge · Edit · Convert · Compress

Clyro is a private web app for PDFs that runs on your own device. Merge and Compress are ready; Edit and Convert
are planned. Nothing is turned into pictures and nothing leaves your device:

- **Compress** shrinks a PDF to a size you choose (default 100 KB). Text stays real, selectable text, links keep
  their exact URLs, and vector artwork stays vector.
- **Merge** combines several PDFs into one. Pages are copied as they are, so nothing is re-encoded; links and
  bookmarks are kept, and you can pick pages from each file (`1-3, 5`).

```bash
./run.sh          # first run installs dependencies, then opens http://127.0.0.1:5050
```

Drop a PDF on the page, set the target size if you want something other than 100 KB, download the result. For
Merge, add your PDFs, drag them into order, optionally choose pages, and press Merge. "Compress this file" and
"Add to Merge" pass a result from one tool to the other. Everything runs on your device; nothing is uploaded
anywhere. The server restarts itself when you edit the code.

The page has a liquid-glass look with a light and a dark theme (sun / moon switch in the top bar). It follows your system setting until you
choose one, then remembers your choice. Colours: peach `#EDC298` for main actions, cyan `#38D6F5` for downloads and secondary actions, teal `#04DAC2`
for success ticks. The logo and icons live in `static/`.

## Why a Figma export is so big

Figma can draw every letter as a filled path with six-decimal coordinates, so one "e" is stored hundreds of
times, and then adds an invisible text layer on top so the text can still be selected. That is most of the
file. Images exported at full resolution and unused font data are the other usual causes.

## What it does

You choose a target size; the tool changes the file **as little as it can** to get there, so the result lands
close to your number (usually 94-99% of it) at the best quality that size allows.

1. **Cleans up**: drops unused font data and hinting, removes duplicate objects, recompresses streams. Lossless.
   If that alone is under your size, you are done.
2. **Vectors**: rounds coordinates (invisible), then, for text drawn as outlines (the Figma case above),
   converts only as much of it as your size needs into a shared embedded font. Letters are put back exactly where
   they were; the rest keeps its exact original outlines. A smaller target converts more, a bigger one less.
3. **Images**, only if still over: picks the highest image quality that fits, from the original images
   untouched up to heavier resizing. Transparency is kept.
4. **Checks its own work**: extracted text and every link must match the original, and rewritten vector pages
   are rendered and compared with the original; missing or extra ink rejects the change.

If the target can't be reached without visibly hurting quality, you get the best quality-preserving result and
a note saying why.

## Files

| File | Purpose |
|---|---|
| `app.py` | Local web server (127.0.0.1 only) |
| `static/index.html` | The page |
| `compressor.py` | Compression pipeline and verification |
| `merger.py` | Merge engine (page selection, bookmarks, thumbnails) |
| `vectorpack.py` | Outlined-text rebuild, content-stream tools |

## Limits

- Password-protected PDFs must be unlocked first.
- Outlined text is only rebuilt when the file also contains the invisible text layer (Figma exports do). Without
  it, nothing says which letter is which, so such files are only shrunk by cleanup and coordinate rounding.
- Rotated text, and text inside form XObjects, are left as they are.
