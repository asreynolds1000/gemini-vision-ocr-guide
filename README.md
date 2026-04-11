# Gemini Vision OCR Guide

Patterns for using Gemini's vision capabilities: OCR/transcription, image analysis, photo cataloging, document extraction. Gemini is currently the strongest model for image understanding tasks — use it over Claude for vision-heavy work.

---

## When to Use Gemini Vision (vs Claude)

| Task | Use Gemini | Use Claude |
|------|-----------|------------|
| Handwriting transcription / OCR | Yes (3.1 Pro) | No — weaker on handwriting |
| Photo description & cataloging | Yes (Flash) | Comparable, but Gemini is faster/cheaper |
| Document text extraction (PDFs, scans) | Yes | Comparable |
| Image classification / tagging | Yes (Flash) | Comparable |
| Multi-page document analysis | Yes (handles 5+ pages) | Limited context for images |
| Code generation from screenshots | Either | Either |

**Rule of thumb:** If the task is primarily "look at this image and tell me what's in it," use Gemini. If the task is "look at this image and then reason about it in a complex way," either works.

---

## Model Selection

| Model | Best For | Daily Limit | Cost |
|-------|---------|-------------|------|
| Gemini 3.1 Pro Preview | Handwriting, difficult OCR, accuracy-critical | 250/day | Higher |
| Gemini 2.5 Pro | Fallback for timeouts, general documents | None | Medium |
| Gemini 2.5 Flash | Photo analysis, cataloging, quick lookups | None | Cheapest |

- **3.1 Pro:** 10-15% more complete output than 2.5 Pro on handwriting. Use for production OCR.
- **Flash:** Failed on 3/4 handwriting test pages, but excellent for photo description/cataloging where precision matters less than coverage.
- **2.5 Pro:** Good all-rounder. Use as fallback when 3.1 Pro times out.

## API Parameters

```python
generation_config = {
    "maxOutputTokens": 65536,   # CRITICAL — default 8192 silently truncates
    "temperature": 0.1,          # For accuracy work. Use 0.3-0.5 for creative descriptions
    "thinkingBudget": 128,       # For 3.1 Pro only; larger budgets don't help
}
```

**maxOutputTokens = 65536:** The single most important setting. Default is 8192, which silently truncates long outputs. This caused 145 truncated files in the diary project's first attempt. Always set explicitly.

**temperature:** 0.1 for OCR/accuracy work (reduces variance in proper nouns by 10-15%). 0.3-0.5 for descriptive tasks where some creativity is fine.

---

## Part 1: OCR & Handwriting Transcription

*Tested on a 506-page microfilm transcription project, March 2026*

### Image Preparation
- **PNG over JPEG** — compression artifacts hurt handwriting recognition
- **Base64 inline** with `mimeType: "image/png"`
- **~1800x1600px at ~2.8MB** worked well. Higher resolution helps but increases payload
- **Full pages, not splits** — splitting 2-page spreads produces worse results than structural prompting on the full image

### Prompt Pattern for OCR

```
[Task description with source context]
[Spelling preservation rules with examples]
[Markup instructions for manuscript features]
[Domain-specific vocabulary list — 15-30 terms]
[Output template with expected structure]
"Output ONLY the transcription."
```

**Key techniques:**
1. **Vocabulary list = ~15-20% accuracy gain** on domain terms. Include proper names, place names, archaic spellings.
2. **Explicitly forbid spelling normalization.** LLMs auto-correct. Say "Preserve spelling exactly" with examples.
3. **Provide output structure** — templates like `## Page NNN` / `### Left page` guide multi-page parsing.
4. **"Output ONLY the transcription"** prevents meta-commentary and apologies.
5. **Markup for manuscript features:** `[word?]`, `[illegible]`, `~~strikethrough~~`, `^insertion^`, `[margin: ...]` — model respects these when instructed.

### Quality Validation Pipeline

**Phase 1: Automated corpus check** (no API calls, <1 min)
- Character count outliers (mean ± 2 SD)
- Short files (<200 chars) = likely truncation
- Hapax clustering (20+ unique words per page) = possible misreads
- ~15-20% of pages flagged for review

**Phase 2: Ground-truth validation** (if published references exist)
- Compare against 15-25 known quotes from historians
- Expected: 85-90% match rate
- Common failures: proper names (~50% error rate), page boundary text

**Phase 3: Targeted human review**
- Review only flagged pages with side-by-side tool (original image + editable text)
- Prioritize: ground-truth discrepancies > short pages > hapax clusters > long pages

### Known OCR Failure Modes

| Failure | Frequency | Mitigation |
|---------|-----------|------------|
| Proper name misreads | ~5-10% | Vocab list helps but doesn't eliminate |
| Spelling normalization | Common | Forbid explicitly with examples |
| Gutter shadow truncation | ~3% | Flag short pages |
| Uncertainty markers omitted | Occasional | Model guesses or skips vs marking [?] |
| Model refusals on historical content | 0/506 pages | Not an issue for Gemini |
| **RECITATION block** (copyrighted books) | ~0.5% (3/549 on Huff) | Both Flash and Pro return empty with `finishReason: RECITATION`. Manual transcription needed. No prompt workaround exists. |

### Phone Scans vs Microfilm (Huff project lesson)

Phone scans (Adobe Scan, etc.) introduce issues microfilm doesn't have:
- **Variable quality:** Lighting, angle, and shadow differ per page. Budget a quality gate before batch transcription — sample 30 pages and grade 1-3.
- **Fingers in frame:** Common at page edges. Models handle this well but captions near edges may be cut off.
- **Duplicate pages at scan boundaries:** When scanning in multiple sessions, the last page of one PDF often equals the first page of the next. **Detect via OCR text comparison, not pixel hash** — the same page scanned twice will have different pixel values.
- **Flash performs identically to Pro on typeset text.** Tested on 5 pages: character-for-character identical output on body text, footnotes, and index pages. Only difference: Pro adds image descriptions on illustration plates.

---

## Part 2: Image Analysis & Photo Cataloging

*Tested on a 153-image historical photo catalog, March 2026*

### Context-Grounded Analysis Prompt

The key technique: **inject existing metadata into the prompt** so Gemini validates/corrects rather than hallucinating from scratch.

```python
prompt = f"""You are analyzing a historical image from [archive name].

CATALOG METADATA:
- Title: {title}
- Type: {item_type}
- Date: {date}
- Era: {era}
- Description: {description}
- Notes: {notes}

TASK:
1. CONFIRM or CORRECT the identification. Does the image match the title/description?
2. READ any visible text (captions, postmarks, handwriting, labels)
3. Note ARCHITECTURAL DETAILS, era indicators, condition clues
4. Identify PEOPLE, vehicles, or notable details
5. DATE CONFIDENCE: Rate high/medium/low; suggest corrections if needed
6. For documents: extract key names, dates, property descriptions

Be concise. Use bullet points. Focus on details NOT already in the catalog."""
```

**Why this works:** Context injection grounds the model in known facts. Instead of "describe this photo" (open-ended, prone to hallucination), you get "does this match what we think it is, and what else can you see?" Much more useful output.

### Model Choice for Cataloging

Use **Gemini 2.5 Flash** — fast, cheap, no daily limit. Quality is sufficient for:
- Photo description and alt-text generation
- Confirming/correcting catalog metadata
- Reading text in photos (signs, labels, postmarks)
- Date estimation from visual clues

Reserve 3.1 Pro for handwriting OCR only.

### Multi-Format Image Handling

**Detect actual file type with magic bytes, not extensions:**
```python
def detect_file_type(path):
    with open(path, "rb") as f:
        header = f.read(8)
    if header[:4] == b"%PDF": return "pdf"
    if header[:3] == b"\xff\xd8\xff": return "jpeg"
    if header[:8] == b"\x89PNG\r\n\x1a\n": return "png"
    if header[:4] in (b"GIF8",): return "gif"
    if header[:4] in (b"II\x2a\x00", b"MM\x00\x2a"): return "tiff"
    return "unknown"
```

**PDFs → images for vision analysis:**
```python
# IMPORTANT: Extract page-by-page, NOT whole PDF at once.
# pdftoppm on a 200+ page PDF at 300 DPI can take 10+ minutes and will
# timeout in batch scripts. Page-by-page also enables skip-existing for resumability.
def convert_pdf_to_images(pdf_path, output_dir, start_seq=1):
    from subprocess import run
    import subprocess
    pages = int(run(["pdfinfo", str(pdf_path)], capture_output=True, text=True)
                .stdout.split("Pages:")[1].split()[0])
    for p in range(1, pages + 1):
        target = output_dir / f"page-{start_seq + p - 1:03d}.png"
        if target.exists():
            continue  # resumable
        prefix = output_dir / "temp"
        run(["pdftoppm", "-f", str(p), "-l", str(p), "-r", "300", "-png",
             str(pdf_path), str(prefix)], timeout=120)
        for tmp in sorted(output_dir.glob("temp-*.png")):
            tmp.rename(target)
            break
```
Send up to 5 pages per request for multi-page PDFs in the catalog workflow.

### Google Sheets Integration Pattern

Read catalog → analyze images → write results back:
```python
# Read: use google-workspace MCP read_sheet_values tool
# Write: use google-workspace MCP modify_sheet_values tool
# Target a specific column (e.g., "gemini_analysis") per row
```

**Useful flags for batch scripts:**
- `--dry-run` — analyze but don't write to sheet (preview mode)
- `--force` — overwrite existing analysis
- `--rows 5,12,38` — selective row processing

### Output → Structured Data

Gemini analysis output feeds into downstream scripts that generate:
- **Alt text** for web images
- **Event mappings** (which historical event does this photo belong to?)
- **Captions** with corrected metadata
- **Validation flags** where Gemini disagrees with catalog metadata

---

## Part 3: Batch Processing (Common to All Tasks)

### Rate Limits (March 2026)
- 3.1 Pro Preview: **250/day**, 1000 RPM
- 2.5 Pro / 2.5 Flash: No daily limit, 1000 RPM

### Batch Runner Pattern
```
1. Check for existing output (auto-skip for resumability)
2. Validate source files exist before submitting
3. Submit in batches of 6-10 with 3-5 parallel workers
4. 1-2 second sleep between batches
5. Collect results as they complete (don't wait for full batch)
6. Write output immediately per-item (not at end)
7. Update progress tracker after each batch
8. On 429 / RESOURCE_EXHAUSTED: print resume command and stop
```

### Retry Strategy
- 3 retries, exponential backoff: `wait = (attempt + 1) * 15` seconds
- Persistent timeout on single item → fall back to 2.5 Pro
- Daily limit hit → stop gracefully, print exact resume command
- **503 UNAVAILABLE** (Flash under high demand) → same backoff; transient, resolves on retry

### Resumability
- Per-item output files (not a single accumulated file)
- Check for existing output at startup → skip completed work
- Progress file: `NEXT_ITEM: NNN` / `STATUS: IN_PROGRESS`
- After crash: re-run same command, picks up where it left off

### Metadata in Every Output
```html
<!-- model: gemini-3.1-pro-preview | date: 2026-03-14 | chars: 3803 -->
```
Enables auditing, dedup detection, and character count analysis without parsing content.

---

## Cost Estimation (March 2026)

| Project Size | Model | Estimated Cost | Wall-Clock Time |
|-------------|-------|---------------|-----------------|
| 500 pages handwriting OCR | 3.1 Pro | ~$1-2 | 2-3 days (rate-limited) |
| 550 pages typeset OCR | 2.5 Flash | ~$0.20 | **24 minutes** (~32 pages/min) |
| 150 photo catalog | 2.5 Flash | ~$0.50 | 1-2 hours |
| 50 document extraction | 2.5 Pro | ~$0.25 | 30 minutes |

Vision work with Gemini is remarkably cheap. For typeset books, Flash is the clear choice — identical quality to Pro at 20x lower cost and no daily rate limit.

---

## Efficiency Lessons (Huff Project, March 2026)

1. **Don't OCR metadata you already know.** We built a per-page Flash pass to detect printed page numbers, then killed it — the PDF-to-book-page offsets were already known from manual inspection. Build manifests from known offsets, validate with 5 spot-checks.

2. **Typeset books are dramatically easier than handwriting.** Mean char count 2338 ± 807, zero long outliers, zero refusals. Quality checking is a formality. Budget 30 min total for a 500-page typeset book vs 3 days for handwriting.

3. **Use text for downstream extraction, not images.** Timeline/event extraction from compiled chapter text (no vision tokens) costs ~1% of re-processing images. Compile first, then extract.

4. **Expect RECITATION blocks on copyrighted books.** Gemini refuses verbatim transcription of some pages (~0.5%). Both Flash and Pro affected. No prompt workaround — manual transcription needed for those pages.

5. **Typeset quality gate can be lightweight.** For clean phone scans of typeset books, sampling 4-5 pages across content types is sufficient. Don't over-engineer the gate.

---

## Structured Output from Gemini (YAML/JSON)

When asking Gemini to output YAML or JSON, expect broken syntax ~30-50% of the time on complex extractions. Three defenses:

1. **Prompt:** Always include "Wrap all string values in double quotes. Escape internal quotes with backslash." This cuts failures by ~70%.

2. **Post-processing `fix_yaml()` function:** Before parsing, scan for unquoted values containing `"`, `: `, `*`, or `(` and wrap them. Catches most remaining failures.

3. **Prefer JSON over YAML** for Gemini output. JSON is stricter but Gemini handles it more reliably — fewer ambiguous syntax edge cases. Parse with `json.loads()` which gives clearer error messages. Convert to YAML after parsing if needed.

```python
# Safer: ask for JSON, convert to YAML
prompt += "\nOutput as a JSON array. Wrap ALL string values in double quotes."

import json, yaml
data = json.loads(raw_text)  # stricter parsing, better errors
yaml.dump(data, file)        # convert after
```

If you must use YAML output, always implement salvage logic — parse up to the first error and keep what worked:
```python
except yaml.YAMLError:
    last_entry = raw_text.rfind("\n- ")
    if last_entry > 0:
        partial = yaml.safe_load(raw_text[:last_entry])
```

---

## Setup & Dependencies

```bash
# Python packages
pip install google-genai google-api-python-client

# System tools (for PDF handling)
brew install poppler  # provides pdftoppm

# API key
export GOOGLE_API_KEY="..."  # or set in .env
```

**Python client:**
```python
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[image_part, prompt],
    config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=65536),
)
```

---

## Useful Script Patterns

| Script Pattern | What It Does |
|--------|-------------|
| Batch transcription | OCR with retry/resume/parallel processing |
| Corpus validation | Automated quality checks (no API calls needed) |
| Image catalog | Photo analysis with existing metadata context injection |
| Structured extraction | Generate structured output from Gemini analysis |

---

## License

CC-BY-4.0
