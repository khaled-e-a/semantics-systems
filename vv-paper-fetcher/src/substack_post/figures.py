"""Find, download and normalize each KEEP paper's main diagram.

Per paper, the first step that works wins:

1. arXiv HTML (https://arxiv.org/html/<id>vN): collect the top-level
   `figure.ltx_figure` elements (subfigure panels, tables and page logos are
   skipped) with their caption, figure number and media: an `<img>`, an
   `<object type="image/svg+xml">`, or an inline TikZ `svg.ltx_picture`.
2. Pick the main diagram: one batched `choose` call (an LLM in production)
   gets every paper's captions; if it fails or returns nonsense, fall back to
   the first "overview/architecture/pipeline/..." caption, then Figure 1.
3. Download the chosen figure. PNG/JPEG are used as-is; SVG is rasterized
   with cairosvg (needs the system cairo library, imported lazily) or, when
   that is unavailable, PyMuPDF. TikZ figures have no file and go to step 4.
4. PDF fallback (no HTML, or the chosen figure is unusable): find the
   "Figure N" / "Fig. N" caption block in the PDF, bound the graphics above
   it, and render that area at 200 dpi.
5. Normalize with Pillow: at most 1456 px wide, transparency flattened onto
   white, PNG when it fits in 900 KB, else JPEG (quality 85, then 70).

One paper's failure never stops the others; it just has no figure.
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pymupdf
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from PIL import Image

from .models import DigestEntry, Figure, PaperDetails
from .paper_details import USER_AGENT, Throttle, is_open_license, new_session

logger = logging.getLogger(__name__)

CaptionChooser = Callable[[Dict[str, List[str]]], Dict[str, int]]

HTML_BASE = "https://arxiv.org/html/"
PDF_BASE = "https://arxiv.org/pdf/"
TIMEOUT_S = 60
FETCH_DELAY_S = 1.0  # between arxiv.org HTML / image / PDF requests

MAX_WIDTH_PX = 1456
MAX_BYTES = 900 * 1024
JPEG_QUALITIES = (85, 70)
PDF_DPI = 200
MIN_SIDE_PX = 64  # smaller downloaded images are icons, not diagrams

MAIN_DIAGRAM_RE = re.compile(r"overview|architecture|pipeline|framework|workflow|illustration", re.IGNORECASE)

_fetch_throttle = Throttle(FETCH_DELAY_S)

_HTML_SRC_WITH_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?/")
_FIG_TAG_NUMBER_RE = re.compile(r"(?:Figure|Fig\.?)\s*([A-Za-z]?\d+(?:\.\d+)*)", re.IGNORECASE)
_ANY_NUMBER_RE = re.compile(r"([A-Za-z]?\d+(?:\.\d+)*)")
# Any figure/table caption start; used to bound the search area above a caption.
_ANY_CAPTION_RE = re.compile(r"^(?:Figure|Fig\.?|Table)\s*[A-Za-z]?\d", re.IGNORECASE)


@dataclass
class HtmlFigure:
    """One top-level figure from an arXiv HTML page."""

    number: str  # "1", "A2", ...; the ordinal when the page has no figure tag
    caption: str  # whitespace-collapsed figcaption text, including "Figure N:"
    kind: str  # "img" | "svg" | "tikz"
    src: Optional[str]  # raw src/data attribute; None for inline TikZ


# ---------------------------------------------------------------------------
# HTML figure extraction
# ---------------------------------------------------------------------------


def _pixel_area(tag: Tag) -> float:
    try:
        return float(tag.get("width", 0)) * float(tag.get("height", 0))
    except (TypeError, ValueError):
        return 0.0


def _pick_panel(candidates: List[Tag], has_flex: bool) -> Tag:
    """First media element, or for a subfigure grid the largest panel (first on ties)."""
    if has_flex and len(candidates) > 1:
        return max(candidates, key=_pixel_area)
    return candidates[0]


def _figure_media(fig: Tag) -> Tuple[Optional[str], Optional[str]]:
    has_flex = fig.find("div", class_="ltx_flex_figure") is not None
    imgs = [i for i in fig.find_all("img") if "ltx_graphics" in (i.get("class") or []) and i.get("src")]
    if imgs:
        return "img", _pick_panel(imgs, has_flex)["src"]
    objects = [
        o for o in fig.find_all("object") if (o.get("type") or "").lower() == "image/svg+xml" and o.get("data")
    ]
    if objects:
        return "svg", _pick_panel(objects, has_flex)["data"]
    if fig.find("svg", class_="ltx_picture") is not None:
        return "tikz", None
    return None, None


def _own_caption(fig: Tag) -> Optional[Tag]:
    """The figure's own figcaption, never one belonging to a nested subfigure/table."""
    own = [fc for fc in fig.find_all("figcaption") if fc.find_parent("figure") is fig]
    for fc in own:
        if fc.find("span", class_="ltx_tag_figure") is not None:
            return fc
    return own[0] if own else None


def _caption_text(caption: Tag) -> str:
    for math in caption.find_all("math"):
        math.replace_with(math.get("alttext", ""))
    return " ".join(caption.get_text().split())


def _figure_number(caption: Optional[Tag]) -> Optional[str]:
    if caption is None:
        return None
    tag = caption.find("span", class_="ltx_tag_figure")
    if tag is None:
        return None
    text = " ".join(tag.get_text().split())
    m = _FIG_TAG_NUMBER_RE.search(text) or _ANY_NUMBER_RE.search(text)
    return m.group(1) if m else None


def extract_html_figures(html: str) -> List[HtmlFigure]:
    """Return the top-level figures of an arXiv HTML page that carry an image, SVG or TikZ."""
    soup = BeautifulSoup(html, "html.parser")
    figures: List[HtmlFigure] = []
    for fig in soup.select("figure.ltx_figure"):
        if fig.find_parent("figure") is not None:  # subfigure panel, or inside a table
            continue
        kind, src = _figure_media(fig)
        if kind is None:  # e.g. a figure environment holding only tables
            continue
        caption = _own_caption(fig)
        number = _figure_number(caption) or str(len(figures) + 1)
        text = _caption_text(caption) if caption is not None else ""
        figures.append(HtmlFigure(number=number, caption=text, kind=kind, src=src))
    return figures


def absolute_media_url(src: str, arxiv_id: str, version: Optional[str]) -> str:
    """Build the absolute URL of a figure file referenced from an arXiv HTML page.

    Current pages use "2609.28614v1/fig.png", which lives at
    "https://arxiv.org/html/" + src. We do not urljoin against the page URL:
    with a trailing slash that doubles the "<id>vN/" segment.
    """
    src = src.strip()
    if re.match(r"^https?://", src, re.IGNORECASE):
        return src
    if src.startswith("//"):
        return "https:" + src
    if src.startswith("/"):
        return "https://arxiv.org" + src
    if _HTML_SRC_WITH_ID_RE.match(src):
        return HTML_BASE + src
    return f"{HTML_BASE}{arxiv_id}{version or ''}/{src}"  # older pages: "x1.png"


# ---------------------------------------------------------------------------
# Choosing the main diagram
# ---------------------------------------------------------------------------


def heuristic_index(figures: List[HtmlFigure]) -> int:
    """First overview/architecture/... caption, else Figure 1, else the first figure."""
    for i, fig in enumerate(figures):
        if MAIN_DIAGRAM_RE.search(fig.caption):
            return i
    for i, fig in enumerate(figures):
        if fig.number == "1":
            return i
    return 0


def _as_index(value: object, count: int) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, int) and 0 <= value < count:
        return value
    return None


def choose_figure_indices(
    figures_by_url: Dict[str, List[HtmlFigure]], choose: Optional[CaptionChooser] = None
) -> Dict[str, int]:
    """Call `choose` once for all papers; fall back per paper to heuristic_index()."""
    picked: Dict[str, object] = {}
    captions = {url: [f.caption for f in figs] for url, figs in figures_by_url.items() if figs}
    if choose is not None and captions:
        try:
            picked = dict(choose(captions) or {})
        except Exception as exc:  # noqa: BLE001 - the heuristic is a complete fallback
            logger.warning("Figure chooser failed, using caption heuristic: %s", exc)
            picked = {}

    chosen: Dict[str, int] = {}
    for url, figs in figures_by_url.items():
        if not figs:
            continue
        index = _as_index(picked.get(url), len(figs))
        if index is None:
            if choose is not None:
                logger.info("No valid chooser index for %s; using caption heuristic", url)
            index = heuristic_index(figs)
        chosen[url] = index
    return chosen


# ---------------------------------------------------------------------------
# Images: download, SVG rasterization, normalization
# ---------------------------------------------------------------------------


def _get(session: requests.Session, url: str) -> requests.Response:
    _fetch_throttle.wait()
    return session.get(url, timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT})


def svg_to_png_bytes(svg: bytes) -> bytes:
    """Rasterize SVG with cairosvg when the cairo library is present, else PyMuPDF."""
    try:
        import cairosvg  # lazy: needs the system cairo library (libcairo2 in CI)
    except (ImportError, OSError) as exc:
        logger.debug("cairosvg unavailable (%s); using PyMuPDF for SVG", exc)
    else:
        try:
            return cairosvg.svg2png(bytestring=svg, scale=2.0, background_color="white")
        except Exception as exc:  # noqa: BLE001 - PyMuPDF is a second chance
            logger.warning("cairosvg failed (%s); using PyMuPDF for SVG", exc)

    doc = pymupdf.open(stream=svg, filetype="svg")
    try:
        page = doc[0]
        # Aim for about 2x the final width (at most 3x the SVG's own size) so the downscale stays sharp.
        dpi = int(max(36, min(216, 72 * 2 * MAX_WIDTH_PX / max(page.rect.width, 1.0))))
        return page.get_pixmap(dpi=dpi, alpha=False).tobytes("png")
    finally:
        doc.close()


def _flatten(img: Image.Image) -> Image.Image:
    """Return an RGB image, compositing any transparency onto white."""
    if img.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img if img.mode == "RGB" else img.convert("RGB")


def normalize_image(img: Image.Image, out_dir: Path, arxiv_id: str) -> Tuple[Path, int, int, str]:
    """Save `img` as out_dir/fig-<id>.png (<= 900 KB) or .jpg; return (path, width, height, mime)."""
    img = _flatten(img)
    if img.width > MAX_WIDTH_PX:
        height = max(1, round(img.height * MAX_WIDTH_PX / img.width))
        img = img.resize((MAX_WIDTH_PX, height), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    ext, mime = "png", "image/png"
    if buf.tell() > MAX_BYTES:
        ext, mime = "jpg", "image/jpeg"
        for quality in JPEG_QUALITIES:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            if buf.tell() <= MAX_BYTES:
                break

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fig-{arxiv_id}.{ext}"  # not with_suffix(): the id contains a dot
    stale = out_dir / f"fig-{arxiv_id}.{'jpg' if ext == 'png' else 'png'}"
    if stale.exists():
        stale.unlink()
    path.write_bytes(buf.getvalue())
    return path, img.width, img.height, mime


def _download_image(url: str, kind: str, session: requests.Session) -> Image.Image:
    resp = _get(session, url)
    resp.raise_for_status()
    content_type = (resp.headers.get("Content-Type") or "").lower()
    is_svg = kind == "svg" or "svg" in content_type or url.lower().split("?", 1)[0].endswith(".svg")
    data = svg_to_png_bytes(resp.content) if is_svg else resp.content
    img = Image.open(io.BytesIO(data))
    img.load()
    if img.width < MIN_SIDE_PX or img.height < MIN_SIDE_PX:
        raise ValueError(f"image too small ({img.width}x{img.height})")
    return img


# ---------------------------------------------------------------------------
# PDF fallback
# ---------------------------------------------------------------------------


@dataclass
class _TextBlock:
    """A text block (or the caption part of one) with its non-blank lines."""

    lines: List[Tuple[pymupdf.Rect, str]]

    @property
    def rect(self) -> pymupdf.Rect:
        return _union([r for r, _ in self.lines])

    @property
    def text(self) -> str:
        return "\n".join(t for _, t in self.lines)

    @property
    def alpha_words(self) -> int:
        return sum(1 for w in self.text.split() if sum(c.isalpha() for c in w) >= 2)


def _caption_re(number: str) -> "re.Pattern[str]":
    # "Figure 1:", "Fig. 1.", "FIG 1 |" — never "Figure 10", "Figure 1.2" or "Fig. S1".
    return re.compile(
        r"^(?:Figure|Fig\.?)\s*" + re.escape(number) + r"(?!\d)(?!\.\d)\s*(?P<delim>[:.|])?",
        re.IGNORECASE,
    )


def _text_blocks(page: pymupdf.Page, caption_re: "re.Pattern[str]") -> List[_TextBlock]:
    """Text blocks of a page; a block is split where a line starts a Figure-N caption.

    PyMuPDF sometimes merges diagram labels and the caption below them into
    one block, so the caption would not be at the block start.
    """
    blocks: List[_TextBlock] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        current: List[Tuple[pymupdf.Rect, str]] = []
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if not text:
                continue
            if current and caption_re.match(text):
                blocks.append(_TextBlock(current))
                current = []
            current.append((pymupdf.Rect(line["bbox"]), text))
        if current:
            blocks.append(_TextBlock(current))
    return blocks


def _union(rects: List[pymupdf.Rect]) -> pymupdf.Rect:
    out = pymupdf.Rect(rects[0])
    for r in rects[1:]:
        out |= r
    return out


def _x_overlap(a: pymupdf.Rect, x0: float, x1: float) -> bool:
    return a.x1 > x0 and a.x0 < x1


def _is_paragraph(block: _TextBlock, span_width: float) -> bool:
    """Running text: several real words and at least two lines spanning most of the column."""
    wide_lines = sum(1 for r, _ in block.lines if r.width >= 0.6 * span_width)
    return block.alpha_words >= 10 and wide_lines >= 2


def _column_span(page: pymupdf.Page, caption: pymupdf.Rect, blocks: List[_TextBlock]) -> Tuple[float, float]:
    """Horizontal extent to search: the caption's column on two-column pages, else the page."""
    width = page.rect.width
    mid = width / 2
    narrow = [
        b.rect for b in blocks if 0.3 * width <= b.rect.width < 0.48 * width and _is_paragraph(b, b.rect.width)
    ]
    two_column = any(r.x1 <= mid + 5 for r in narrow) and any(r.x0 >= mid - 5 for r in narrow)
    if two_column and caption.width < 0.55 * width:
        if caption.x1 <= mid + 5:
            return 0.0, mid
        if caption.x0 >= mid - 5:
            return mid, width
    return 0.0, width


def _image_rects(page: pymupdf.Page) -> List[pymupdf.Rect]:
    rects = [pymupdf.Rect(info["bbox"]) for info in page.get_image_info()]
    for img in page.get_images(full=True):
        try:
            rects.extend(pymupdf.Rect(r) for r in page.get_image_rects(img[0]))
        except Exception:  # noqa: BLE001 - odd xrefs just contribute nothing
            continue
    return [r for r in rects if r.is_valid and not r.is_empty]


def _body_top(page_blocks: List[list]) -> Optional[float]:
    """Estimate the top margin of the text body (running headers sit above it).

    Uses a low percentile of each page's topmost paragraph-like block, from
    the plain get_text("blocks") output.
    """
    tops = []
    for blocks in page_blocks:
        ys = [
            b[1]
            for b in blocks
            if b[6] == 0
            and b[4].count("\n") >= 2
            and sum(1 for w in b[4].split() if sum(c.isalpha() for c in w) >= 2) >= 10
        ]
        if ys:
            tops.append(min(ys))
    if not tops:
        return None
    tops.sort()
    return tops[int(0.2 * (len(tops) - 1))]


def _figure_rect_above(
    page: pymupdf.Page,
    caption: _TextBlock,
    blocks: List[_TextBlock],
    drawings: list,
    images: List[pymupdf.Rect],
    body_top: Optional[float] = None,
) -> Optional[pymupdf.Rect]:
    """Bound the graphics between the previous body-text block (or top margin) and the caption top."""
    cap_top = caption.lines[0][0].y0
    x0, x1 = _column_span(page, caption.rect, blocks)
    all_graphics = [d["rect"] for d in drawings] + images

    top = page.rect.y0
    if body_top is not None and body_top - 6 < cap_top - 10:
        top = body_top - 6  # skip running headers/logos in the top margin
    for block in blocks:
        r = block.rect
        if block is caption or not _x_overlap(r, x0, x1) or r.y1 > cap_top + 1:
            continue
        is_body = _is_paragraph(block, x1 - x0) and not any(r.intersects(g) for g in all_graphics)
        is_other_caption = _ANY_CAPTION_RE.match(block.text) is not None
        if is_body or is_other_caption:
            top = max(top, r.y1)

    region = pymupdf.Rect(x0, top, x1, cap_top)
    if region.is_empty or region.height < 10:
        return None

    parts = [r for r in page.cluster_drawings(clip=region, drawings=drawings) if r.width >= 5 and r.height >= 5]
    for img in images:
        clipped = img & region
        if not clipped.is_empty and clipped.width >= 5 and clipped.height >= 5:
            parts.append(clipped)
    if not parts:
        return None
    fig = _union(parts)

    # Pull in diagram labels: text inside the region that touches the graphics.
    for _ in range(2):
        grown = pymupdf.Rect(fig.x0 - 12, fig.y0 - 12, fig.x1 + 12, fig.y1 + 12)
        for block in blocks:
            r = block.rect
            if block is caption or r.y0 < region.y0 - 1 or r.y1 > region.y1 + 1:
                continue
            if r.intersects(grown):
                fig |= r & region
    # A little padding, but never down into the caption itself.
    fig = pymupdf.Rect(fig.x0 - 4, fig.y0 - 4, fig.x1 + 4, min(fig.y1 + 4, cap_top)) & page.rect
    if fig.width < 40 or fig.height < 30:
        return None
    return fig


def find_pdf_figure(doc: pymupdf.Document, number: str) -> Optional[Tuple[int, pymupdf.Rect]]:
    """Locate Figure `number` in a PDF: (page index, clip rect), or None.

    The caption is the text block that starts with "Figure N" / "Fig. N".
    Caption-like matches ("Figure 1:" or "Fig. 1 Overview") are tried before
    body text that merely starts with "Figure 1 shows ...". A caption with no
    graphics above it (e.g. a caption placed above its figure) is skipped.
    """
    pattern = _caption_re(number)
    deferred: List[Tuple[int, _TextBlock, List[_TextBlock]]] = []
    page_blocks = [doc[i].get_text("blocks") for i in range(doc.page_count)]
    body_top = _body_top(page_blocks)

    def try_caption(page_index: int, caption: _TextBlock, blocks: List[_TextBlock]) -> Optional[pymupdf.Rect]:
        page = doc[page_index]
        return _figure_rect_above(page, caption, blocks, page.get_drawings(), _image_rects(page), body_top)

    for page_index in range(doc.page_count):
        page = doc[page_index]
        # Cheap scan of the plain blocks first; the detailed layout is built only for pages with a hit.
        if not any(
            b[6] == 0 and any(pattern.match(line.strip()) for line in b[4].splitlines())
            for b in page_blocks[page_index]
        ):
            continue
        blocks = _text_blocks(page, pattern)
        for block in sorted(blocks, key=lambda b: b.rect.y0):
            m = pattern.match(block.text)
            if not m:
                continue
            rest = block.text[m.end() :].lstrip()
            if not (m.group("delim") or rest[:1].isupper() or rest[:1] == "("):
                deferred.append((page_index, block, blocks))  # "Figure 1 shows ..." body text
                continue
            rect = try_caption(page_index, block, blocks)
            if rect is not None:
                return page_index, rect

    for page_index, block, blocks in deferred:
        rect = try_caption(page_index, block, blocks)
        if rect is not None:
            return page_index, rect
    return None


def render_pdf_figure(pdf_bytes: bytes, number: str) -> Optional[Image.Image]:
    """Render Figure `number` from a PDF at PDF_DPI, or None when it cannot be located."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        found = find_pdf_figure(doc, number)
        if found is None:
            return None
        page_index, rect = found
        pix = doc[page_index].get_pixmap(clip=rect, dpi=PDF_DPI, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        img.load()
        return img
    finally:
        doc.close()


def _pdf_figure(arxiv_id: str, version: str, number: str, session: requests.Session) -> Optional[Image.Image]:
    url = f"{PDF_BASE}{arxiv_id}{version}"
    resp = _get(session, url)
    resp.raise_for_status()
    if not resp.content.startswith(b"%PDF"):
        raise ValueError(f"{url} did not return a PDF")
    return render_pdf_figure(resp.content, number)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _version(entry: DigestEntry, details: Optional[PaperDetails]) -> str:
    return entry.version or (details.version if details else None) or ""


def _html_figures(entry: DigestEntry, version: str, session: requests.Session) -> List[HtmlFigure]:
    url = f"{HTML_BASE}{entry.arxiv_id}{version}"
    resp = _get(session, url)
    if resp.status_code == 404:
        logger.info("No arXiv HTML for %s (404); will use the PDF", entry.arxiv_id)
        return []
    resp.raise_for_status()
    figures = extract_html_figures(resp.text)
    if not figures:
        logger.info("arXiv HTML for %s has no usable figures; will use the PDF", entry.arxiv_id)
    return figures


def _build_figure(
    entry: DigestEntry,
    details: Optional[PaperDetails],
    figures: List[HtmlFigure],
    index: Optional[int],
    out_dir: Path,
    session: requests.Session,
) -> Optional[Figure]:
    arxiv_id = entry.arxiv_id or ""
    version = _version(entry, details)
    number = "1"
    img: Optional[Image.Image] = None
    source = "pdf"

    if figures and index is not None:
        chosen = figures[index]
        number = chosen.number
        if chosen.kind in ("img", "svg") and chosen.src:
            media_url = absolute_media_url(chosen.src, arxiv_id, version)
            try:
                img = _download_image(media_url, chosen.kind, session)
                source = "html"
            except Exception as exc:  # noqa: BLE001 - fall through to the PDF
                logger.warning("Figure %s of %s unusable from HTML (%s); trying the PDF", number, arxiv_id, exc)
        else:
            logger.info("Figure %s of %s is inline TikZ; rendering it from the PDF", number, arxiv_id)

    if img is None:
        img = _pdf_figure(arxiv_id, version, number, session)
        source = "pdf"
    if img is None:
        logger.warning("No figure found for %s (Figure %s not located in the PDF)", arxiv_id, number)
        return None

    path, width, height, mime = normalize_image(img, out_dir, arxiv_id)
    return Figure(
        url=entry.url,
        path=path,
        figure_number=number,
        credit_caption=f"Figure {number} from “{entry.title}” (arXiv:{arxiv_id})",
        source=source,
        openly_licensed=is_open_license(details.license_url if details else None),
        width=width,
        height=height,
        mime=mime,
    )


def fetch_main_figures(
    entries: List[DigestEntry],
    details: Dict[str, PaperDetails],
    out_dir: Path,
    choose: Optional[CaptionChooser] = None,
    session: Optional[requests.Session] = None,
) -> Dict[str, Figure]:
    """Download each entry's main diagram into out_dir; keyed by entry.url.

    Entries without an arXiv id, or with no usable figure, are absent. A
    single paper's failure is logged and never raised.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = session or new_session()
    details = details or {}

    papers: List[DigestEntry] = []
    seen = set()
    for entry in entries:
        if entry.arxiv_id and entry.url not in seen:
            seen.add(entry.url)
            papers.append(entry)

    html_figures: Dict[str, List[HtmlFigure]] = {}
    for entry in papers:
        try:
            figs = _html_figures(entry, _version(entry, details.get(entry.url)), session)
        except Exception as exc:  # noqa: BLE001 - the PDF is still an option
            logger.warning("arXiv HTML fetch failed for %s: %s", entry.arxiv_id, exc)
            figs = []
        if figs:
            html_figures[entry.url] = figs

    chosen = choose_figure_indices(html_figures, choose)

    results: Dict[str, Figure] = {}
    for entry in papers:
        try:
            figure = _build_figure(
                entry,
                details.get(entry.url),
                html_figures.get(entry.url, []),
                chosen.get(entry.url),
                out_dir,
                session,
            )
        except Exception as exc:  # noqa: BLE001 - one paper must never kill the run
            logger.warning("Figure extraction failed for %s: %s", entry.url, exc)
            continue
        if figure is not None:
            results[entry.url] = figure
            logger.info(
                "Figure %s for %s from %s -> %s (%dx%d)",
                figure.figure_number, entry.arxiv_id, figure.source, figure.path.name, figure.width, figure.height,
            )

    logger.info("Found figures for %d/%d papers", len(results), len(papers))
    return results
