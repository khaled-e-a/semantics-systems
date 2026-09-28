import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf
import pytest
import requests
from PIL import Image

from src.substack_post import figures, paper_details
from src.substack_post.figures import (
    HtmlFigure,
    absolute_media_url,
    choose_figure_indices,
    extract_html_figures,
    fetch_main_figures,
    find_pdf_figure,
    heuristic_index,
    normalize_image,
    render_pdf_figure,
    svg_to_png_bytes,
)
from src.substack_post.models import DigestEntry, PaperDetails

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "substack"
HTML_FIXTURE = (FIXTURES / "arxiv_html_2609.28614v1_trimmed.html").read_text(encoding="utf-8")

BODY = (
    "The method section describes how the system is trained and evaluated on held-out data "
    "with several baselines and careful ablations across many different settings. "
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int = 200, content: bytes = b"", content_type: str = "") -> None:
        self.status_code = status_code
        self.content = content
        self.headers = {"Content-Type": content_type} if content_type else {}

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class FakeSession:
    """Serves fixed responses by exact URL; anything else is a 404."""

    def __init__(self, routes: Dict[str, Any]) -> None:
        self.routes = routes
        self.calls: List[str] = []

    def get(self, url: str, **kwargs) -> FakeResponse:
        assert kwargs.get("timeout"), "every request needs a timeout"
        self.calls.append(url)
        result = self.routes.get(url, FakeResponse(404, b"not found"))
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture(autouse=True)
def no_throttle(monkeypatch):
    monkeypatch.setattr(figures._fetch_throttle, "min_interval_s", 0)
    monkeypatch.setattr(paper_details._api_throttle, "min_interval_s", 0)


@pytest.fixture
def no_cairo(monkeypatch):
    """Force the PyMuPDF SVG path, as on a machine without the cairo library."""
    monkeypatch.setitem(sys.modules, "cairosvg", None)


def png_bytes(size=(952, 334), color=(30, 120, 200, 255), mode="RGBA") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


SVG = (
    b'<svg xmlns="http://www.w3.org/2000/svg" width="400pt" height="200pt" viewBox="0 0 400 200">'
    b'<rect x="20" y="20" width="360" height="160" fill="#3366cc"/></svg>'
)


def add_figure_page(doc, caption: str, *, lead: str = BODY * 2, caption_above: bool = False) -> None:
    """A page with a paragraph, a vector diagram with labels, its caption, and more body text."""
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(pymupdf.Rect(72, 72, 540, 140), lead, fontsize=10)
    if caption_above:
        page.insert_textbox(pymupdf.Rect(72, 150, 540, 175), caption, fontsize=9)
        page.draw_rect(pymupdf.Rect(150, 190, 460, 360), color=(0, 0, 0.5), fill=(0.2, 0.4, 0.9))
        page.insert_textbox(pymupdf.Rect(72, 400, 540, 480), BODY * 3, fontsize=10)
        return
    page.draw_rect(pymupdf.Rect(150, 160, 460, 330), color=(0, 0, 0.5), fill=(0.2, 0.4, 0.9))
    page.insert_text((180, 250), "Encoder", fontsize=12)
    page.draw_line((460, 245), (500, 245))
    page.insert_text((505, 248), "out", fontsize=9)  # label outside the box
    page.insert_textbox(pymupdf.Rect(72, 345, 540, 380), caption, fontsize=9)
    page.insert_textbox(pymupdf.Rect(72, 400, 540, 480), BODY * 3, fontsize=10)


def make_pdf(*captions: str, **kwargs) -> bytes:
    doc = pymupdf.open()
    for caption in captions:
        add_figure_page(doc, caption, **kwargs)
    return doc.tobytes()


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------


def test_extract_html_figures_from_captured_page():
    figs = extract_html_figures(HTML_FIXTURE)

    # S1.F1 img, S1.F2 object/svg, S6.F7 subfigure panels, S7.F10 TikZ wrapping an img;
    # tables (figure.ltx_table), page logos and the table-only S5.fig3 are skipped.
    assert [f.number for f in figs] == ["1", "2", "7", "10"]
    assert [f.kind for f in figs] == ["img", "svg", "img", "img"]
    assert figs[0].src == "2609.28614v1/fig_teaser_repacked.png"
    assert figs[1].src == "2609.28614v1/fig_setups.svg"
    assert figs[2].src == "2609.28614v1/figs/fig5_heatmap.png"  # largest panel of the grid
    assert figs[3].src == "2609.28614v1/fig_tax_map.png"
    assert all("logo" not in (f.src or "") and "openai.svg" not in (f.src or "") for f in figs)

    caption = figs[0].caption
    assert caption.startswith("Figure 1: From spontaneous reward hacking to adaptive evasion under oversight.")
    assert "\n" not in caption and "  " not in caption
    assert "Table" not in " ".join(f.caption[:8] for f in figs)


NESTED_AND_TIKZ = """
<html><body>
<img src="/static/browse/0.3.4/images/arxiv-logo-one-color-white.svg" alt="logo">
<figure class="ltx_figure" id="S3.F3">
  <div class="ltx_flex_figure">
    <div class="ltx_flex_cell ltx_flex_size_2">
      <figure class="ltx_figure ltx_figure_panel ltx_align_center" id="S3.F3.sf1">
        <img src="2609.00001v2/panel_a.png" class="ltx_graphics ltx_img_square" width="120" height="100" alt="">
        <figcaption class="ltx_caption"><span class="ltx_tag ltx_tag_figure">(a) </span>left panel</figcaption>
      </figure>
    </div>
    <div class="ltx_flex_cell ltx_flex_size_2">
      <figure class="ltx_figure ltx_figure_panel ltx_align_center" id="S3.F3.sf2">
        <img src="2609.00001v2/panel_b.png" class="ltx_graphics ltx_img_landscape" width="300" height="150" alt="">
        <figcaption class="ltx_caption"><span class="ltx_tag ltx_tag_figure">(b) </span>right panel</figcaption>
      </figure>
    </div>
  </div>
  <figcaption class="ltx_caption ltx_centering"><span class="ltx_tag ltx_tag_figure">Figure 3: </span>Two
     views of the <math alttext="\\alpha" class="ltx_Math"><mi>α</mi></math> sweep.</figcaption>
</figure>
<figure class="ltx_figure" id="S4.F4">
  <svg class="ltx_picture ltx_centering" width="200" height="100"><g><path d="M 0 0 L 10 10"></path></g></svg>
  <figcaption class="ltx_caption"><span class="ltx_tag ltx_tag_figure">Fig. 4. </span>TikZ architecture
    diagram.</figcaption>
</figure>
<figure class="ltx_figure" id="S4.F5">
  <img src="x5.png" class="ltx_graphics" width="200" height="100" alt="">
  <figcaption class="ltx_caption">A caption without a figure tag.</figcaption>
</figure>
<figure class="ltx_table" id="S4.T1">
  <img src="2609.00001v2/table_as_image.png" class="ltx_graphics" width="200" height="100" alt="">
  <figcaption class="ltx_caption"><span class="ltx_tag ltx_tag_table">Table 1: </span>Not a figure.</figcaption>
</figure>
</body></html>
"""


def test_extract_nested_subfigures_tikz_and_untagged():
    figs = extract_html_figures(NESTED_AND_TIKZ)
    assert len(figs) == 3

    nested, tikz, untagged = figs
    assert nested.number == "3" and nested.kind == "img"
    assert nested.src == "2609.00001v2/panel_b.png"  # the largest panel
    assert nested.caption == "Figure 3: Two views of the \\alpha sweep."  # not the (a)/(b) sub-captions

    assert (tikz.number, tikz.kind, tikz.src) == ("4", "tikz", None)
    assert untagged.number == "3"  # ordinal among kept figures
    assert untagged.caption == "A caption without a figure tag."


@pytest.mark.parametrize(
    "src, expected",
    [
        ("2609.28614v1/fig_teaser_repacked.png", "https://arxiv.org/html/2609.28614v1/fig_teaser_repacked.png"),
        ("2609.28614v1/figs/fig1_exp1.png", "https://arxiv.org/html/2609.28614v1/figs/fig1_exp1.png"),
        ("x1.png", "https://arxiv.org/html/2609.28614v1/x1.png"),  # older pages
        ("/static/img/logo.png", "https://arxiv.org/static/img/logo.png"),
        ("https://cdn.example.org/f.png", "https://cdn.example.org/f.png"),
        ("//cdn.example.org/f.png", "https://cdn.example.org/f.png"),
    ],
)
def test_absolute_media_url(src, expected):
    url = absolute_media_url(src, "2609.28614", "v1")
    assert url == expected
    assert "2609.28614v1/2609.28614v1" not in url  # never doubled like urljoin with a trailing slash


# ---------------------------------------------------------------------------
# choosing the main diagram
# ---------------------------------------------------------------------------


def _figs(*captions: str) -> List[HtmlFigure]:
    return [HtmlFigure(number=str(i + 1), caption=c, kind="img", src=f"f{i}.png") for i, c in enumerate(captions)]


def test_heuristic_index():
    assert heuristic_index(_figs("Figure 1: Results.", "Figure 2: System ARCHITECTURE.")) == 1
    assert heuristic_index(_figs("Figure 1: Results.", "Figure 2: More results.")) == 0
    unnumbered = [HtmlFigure("A1", "Plot", "img", "a"), HtmlFigure("1", "Teaser", "img", "b")]
    assert heuristic_index(unnumbered) == 1  # falls back to Figure 1


def test_choose_calls_chooser_once_and_validates():
    by_url = {
        "a": _figs("Figure 1: Results.", "Figure 2: Pipeline overview."),
        "b": _figs("Figure 1: Teaser.", "Figure 2: Ablation."),
        "c": _figs("Figure 1: Workflow of the agent."),
        "d": _figs("Figure 1: X.", "Figure 2: Y."),
        "e": _figs("Figure 1: X.", "Figure 2: Y."),
        "f": _figs("Figure 1: X.", "Figure 2: Y."),
    }
    calls = []

    def chooser(captions: Dict[str, List[str]]) -> Dict[str, int]:
        calls.append(captions)
        return {"a": 0, "b": 1, "c": 5, "d": True, "e": "1", "f": -1}

    chosen = choose_figure_indices(by_url, chooser)

    assert len(calls) == 1
    assert calls[0] == {url: [f.caption for f in figs] for url, figs in by_url.items()}
    assert chosen == {"a": 0, "b": 1, "c": 0, "d": 0, "e": 1, "f": 0}


def test_choose_falls_back_when_chooser_fails_or_missing():
    by_url = {"a": _figs("Figure 1: Results.", "Figure 2: Framework."), "b": _figs("Figure 1: Teaser.")}

    def broken(captions):
        raise RuntimeError("LLM down")

    assert choose_figure_indices(by_url, broken) == {"a": 1, "b": 0}
    assert choose_figure_indices(by_url, None) == {"a": 1, "b": 0}
    assert choose_figure_indices(by_url, lambda captions: None) == {"a": 1, "b": 0}
    assert choose_figure_indices(by_url, lambda captions: {"a": 0}) == {"a": 0, "b": 0}


def test_chooser_not_called_without_figures():
    called = []
    assert choose_figure_indices({}, lambda c: called.append(c) or {}) == {}
    assert called == []


# ---------------------------------------------------------------------------
# PDF fallback
# ---------------------------------------------------------------------------


def test_find_pdf_figure_bounds_diagram_above_caption():
    doc = pymupdf.open(stream=make_pdf("Fig. 1: Overview of the pipeline. The encoder feeds the decoder.",
                                       lead="Figure 1 shows the overall pipeline. " + BODY * 2), filetype="pdf")
    found = find_pdf_figure(doc, "1")
    assert found is not None
    page_index, rect = found
    assert page_index == 0
    # the diagram (150..460 x 160..330) plus the "out" label at x≈505, not the text above or the caption
    assert 140 <= rect.x0 <= 150 and rect.x1 >= 515
    assert 150 <= rect.y0 <= 160 and 330 <= rect.y1 <= 345


def test_render_pdf_figure_produces_image():
    pdf = make_pdf("Figure 1: Overview of the framework.")
    img = render_pdf_figure(pdf, "1")
    assert img is not None
    # ≈ (380 x 178 pt) at 200 dpi
    assert 900 <= img.width <= 1150 and 450 <= img.height <= 540
    r, g, b = img.convert("RGB").getpixel((img.width // 3, img.height // 5))
    assert b > r and b > g  # inside the blue diagram


def test_pdf_figure_number_is_exact():
    pdf = pymupdf.open(stream=make_pdf("Figure 10: Results on the benchmark.", "Fig. S1: Supplementary view."),
                       filetype="pdf")
    assert find_pdf_figure(pdf, "1") is None  # neither "Figure 10" nor "Fig. S1"
    assert find_pdf_figure(pdf, "10")[0] == 0
    assert find_pdf_figure(pdf, "S1")[0] == 1


def test_pdf_caption_above_figure_gives_up():
    pdf = make_pdf("Figure 1: A caption placed above its figure.", caption_above=True)
    assert render_pdf_figure(pdf, "1") is None


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------


def test_normalize_wide_transparent_png(tmp_path):
    img = Image.new("RGBA", (3000, 1000), (0, 0, 0, 0))  # fully transparent
    img.paste((200, 0, 0, 255), (1000, 300, 2000, 700))

    path, width, height, mime = normalize_image(img, tmp_path, "2609.28614")

    assert path == tmp_path / "fig-2609.28614.png"
    assert (width, height, mime) == (1456, 485, "image/png")
    saved = Image.open(path)
    assert saved.size == (1456, 485) and saved.mode == "RGB"
    assert saved.getpixel((5, 5)) == (255, 255, 255)  # transparency flattened onto white
    assert os.path.getsize(path) <= figures.MAX_BYTES


def test_normalize_large_image_becomes_jpeg(tmp_path):
    noisy = Image.frombytes("RGB", (1400, 1000), os.urandom(1400 * 1000 * 3))
    (tmp_path / "fig-2609.00001.png").write_bytes(b"stale")

    path, width, height, mime = normalize_image(noisy, tmp_path, "2609.00001")

    assert path == tmp_path / "fig-2609.00001.jpg"
    assert (width, height, mime) == (1400, 1000, "image/jpeg")
    assert os.path.getsize(path) <= figures.MAX_BYTES
    assert not (tmp_path / "fig-2609.00001.png").exists()
    assert Image.open(path).format == "JPEG"


def test_svg_rasterizes_without_cairo(no_cairo):
    img = Image.open(io.BytesIO(svg_to_png_bytes(SVG)))
    assert img.format == "PNG" and img.width >= 400
    r, g, b = img.convert("RGB").getpixel((img.width // 2, img.height // 2))
    assert b > r


def test_svg_prefers_cairosvg_when_available(monkeypatch):
    calls = []

    class FakeCairo:
        @staticmethod
        def svg2png(bytestring, **kwargs):
            calls.append(kwargs)
            return png_bytes((800, 400))

    monkeypatch.setitem(sys.modules, "cairosvg", FakeCairo)
    assert Image.open(io.BytesIO(svg_to_png_bytes(SVG))).size == (800, 400)
    assert len(calls) == 1


def test_svg_falls_back_when_cairosvg_errors(monkeypatch):
    class BrokenCairo:
        @staticmethod
        def svg2png(bytestring, **kwargs):
            raise ValueError("unsupported element")

    monkeypatch.setitem(sys.modules, "cairosvg", BrokenCairo)
    assert Image.open(io.BytesIO(svg_to_png_bytes(SVG))).format == "PNG"


# ---------------------------------------------------------------------------
# end to end with a mocked session
# ---------------------------------------------------------------------------

URL_HTML = "http://arxiv.org/abs/2609.28614v1"
URL_PDF_ONLY = "http://arxiv.org/abs/2609.30489v1"
URL_BROKEN = "http://arxiv.org/abs/2609.29935v1"
URL_HF = "https://huggingface.co/papers/2609.11115"
URL_NO_ID = "https://openreview.net/forum?id=AbC123"


def _entry(rank: int, url: str, arxiv_id: Optional[str], version: Optional[str], title: str) -> DigestEntry:
    return DigestEntry(rank=rank, section="top", title=title, url=url, arxiv_id=arxiv_id, version=version)


def _details(arxiv_id: str, version: str, license_url: Optional[str] = None) -> PaperDetails:
    return PaperDetails(arxiv_id=arxiv_id, version=version, title="T", abstract="A", authors=["X"],
                        license_url=license_url)


def test_fetch_main_figures_end_to_end(tmp_path):
    entries = [
        _entry(1, URL_HTML, "2609.28614", "v1", "Reward Hacking Challenges Oversight of Autonomous Research Agents"),
        _entry(2, URL_PDF_ONLY, "2609.30489", "v1", "BioEVAL"),
        _entry(3, URL_BROKEN, "2609.29935", "v1", "Robust Detection"),
        _entry(4, URL_NO_ID, None, None, "No arXiv id"),
    ]
    details = {URL_HTML: _details("2609.28614", "v1", "http://creativecommons.org/licenses/by/4.0/")}
    session = FakeSession(
        {
            "https://arxiv.org/html/2609.28614v1": FakeResponse(200, HTML_FIXTURE.encode("utf-8"), "text/html"),
            "https://arxiv.org/html/2609.28614v1/fig_teaser_repacked.png": FakeResponse(200, png_bytes(), "image/png"),
            "https://arxiv.org/html/2609.30489v1": FakeResponse(404, b"No HTML for '2609.30489v1'"),
            "https://arxiv.org/pdf/2609.30489v1": FakeResponse(
                200, make_pdf("Fig. 1 Overview of the BioEVAL benchmarking framework."), "application/pdf"
            ),
            "https://arxiv.org/html/2609.29935v1": requests.ConnectionError("reset"),
            "https://arxiv.org/pdf/2609.29935v1": FakeResponse(500, b"oops"),
        }
    )
    chooser_calls = []

    def chooser(captions):
        chooser_calls.append(captions)
        return {URL_HTML: 0}

    result = fetch_main_figures(entries, details, tmp_path / "figs", choose=chooser, session=session)

    assert set(result) == {URL_HTML, URL_PDF_ONLY}
    assert len(chooser_calls) == 1 and list(chooser_calls[0]) == [URL_HTML]
    assert len(chooser_calls[0][URL_HTML]) == 4

    html_fig = result[URL_HTML]
    assert html_fig.source == "html" and html_fig.figure_number == "1"
    assert html_fig.credit_caption == (
        "Figure 1 from “Reward Hacking Challenges Oversight of Autonomous Research Agents” (arXiv:2609.28614)"
    )
    assert html_fig.path == tmp_path / "figs" / "fig-2609.28614.png" and html_fig.path.exists()
    assert (html_fig.width, html_fig.height, html_fig.mime) == (952, 334, "image/png")
    assert html_fig.openly_licensed is True
    assert html_fig.url == URL_HTML

    pdf_fig = result[URL_PDF_ONLY]
    assert pdf_fig.source == "pdf" and pdf_fig.figure_number == "1"
    assert pdf_fig.credit_caption == "Figure 1 from “BioEVAL” (arXiv:2609.30489)"
    assert pdf_fig.openly_licensed is False  # no details → not known to be open
    assert pdf_fig.path.exists() and pdf_fig.mime in ("image/png", "image/jpeg")
    assert not any("openreview" in url for url in session.calls)


def test_svg_choice_and_pdf_fallback_for_unusable_or_tikz(tmp_path, no_cairo):
    tikz_html = NESTED_AND_TIKZ.encode("utf-8")
    other_html = HTML_FIXTURE.replace("2609.28614v1/", "2609.29935v1/").encode("utf-8")
    entries = [
        _entry(1, URL_HTML, "2609.28614", "v1", "Svg Paper"),
        _entry(2, URL_BROKEN, "2609.29935", "v1", "Broken Image Paper"),
        _entry(3, URL_HF, "2609.11115", None, "TikZ Paper"),
    ]
    details = {URL_HF: _details("2609.11115", "v3")}  # HF url: version comes from arXiv details
    session = FakeSession(
        {
            "https://arxiv.org/html/2609.28614v1": FakeResponse(200, HTML_FIXTURE.encode("utf-8")),
            "https://arxiv.org/html/2609.28614v1/fig_setups.svg": FakeResponse(200, SVG, "image/svg+xml"),
            "https://arxiv.org/html/2609.29935v1": FakeResponse(200, other_html),
            # 2609.29935v1/fig_setups.svg is not routed → 404 → PDF fallback for Figure 2
            "https://arxiv.org/pdf/2609.29935v1": FakeResponse(200, make_pdf("Figure 2: Evaluation settings.")),
            "https://arxiv.org/html/2609.11115v3": FakeResponse(200, tikz_html),
            "https://arxiv.org/pdf/2609.11115v3": FakeResponse(200, make_pdf("Fig. 4. TikZ architecture diagram.")),
        }
    )

    def chooser(captions):
        return {URL_HTML: 1, URL_BROKEN: 1, URL_HF: 1}  # Figure 2 (svg), Figure 2 (svg), Fig. 4 (TikZ)

    result = fetch_main_figures(entries, details, tmp_path, choose=chooser, session=session)

    assert result[URL_HTML].source == "html" and result[URL_HTML].figure_number == "2"
    assert result[URL_HTML].mime == "image/png" and result[URL_HTML].width >= 400

    assert result[URL_BROKEN].source == "pdf" and result[URL_BROKEN].figure_number == "2"
    assert "https://arxiv.org/html/2609.29935v1/fig_setups.svg" in session.calls

    assert result[URL_HF].source == "pdf" and result[URL_HF].figure_number == "4"
    assert result[URL_HF].credit_caption == "Figure 4 from “TikZ Paper” (arXiv:2609.11115)"


def test_one_paper_failure_never_raises(tmp_path, monkeypatch):
    entries = [_entry(1, URL_HTML, "2609.28614", "v1", "T")]

    def boom(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(figures, "normalize_image", boom)
    session = FakeSession(
        {
            "https://arxiv.org/html/2609.28614v1": FakeResponse(200, HTML_FIXTURE.encode("utf-8")),
            "https://arxiv.org/html/2609.28614v1/fig_teaser_repacked.png": FakeResponse(200, png_bytes()),
        }
    )
    assert fetch_main_figures(entries, {}, tmp_path, session=session) == {}
