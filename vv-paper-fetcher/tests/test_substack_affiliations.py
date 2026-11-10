"""affiliations.py: author-block extraction (HTML/PDF) and LLM classification, all offline."""
import json
from pathlib import Path
from typing import Any, Dict, List, Union
from unittest.mock import MagicMock

import pymupdf
import pytest
import requests

from src.substack_post import affiliations, figures, paper_details
from src.substack_post.affiliations import (
    MAX_BLOCK_CHARS,
    Affiliations,
    classify_affiliations,
    extract_html_author_block,
    extract_pdf_author_block,
    fetch_author_blocks,
    has_institution_words,
    lookup_affiliations,
)
from src.substack_post.models import DigestEntry, PaperDetails
from src.substack_post.prompts import JSON_RETRY_PREFIX

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "substack"


def _fixture(arxiv_id: str) -> str:
    return (FIXTURES / f"arxiv_html_{arxiv_id}_authors_trimmed.html").read_text(encoding="utf-8")


TRUSTED = {
    "tier1": ["MIT", "Stanford University", "Carnegie Mellon University", "Google DeepMind"],
    "tier2": ["University of Edinburgh", "Tsinghua University"],
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int = 200, content: bytes = b"") -> None:
        self.status_code = status_code
        self.content = content
        self.headers: Dict[str, str] = {}

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


def _resp(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    return resp


def _client(responses: List[Union[str, Exception]]):
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        r if isinstance(r, Exception) else _resp(r) for r in responses
    ]
    return client


def _messages(client, call_index: int):
    return client.chat.completions.create.call_args_list[call_index].kwargs["messages"]


def _results(*items) -> str:
    return json.dumps(
        {"results": [{"index": i, "institutions": inst, "matched": matched} for i, inst, matched in items]}
    )


def _entry(arxiv_id, version="v1", url=None) -> DigestEntry:
    return DigestEntry(
        rank=1,
        section="top",
        title=f"Paper {arxiv_id}",
        url=url or (f"http://arxiv.org/abs/{arxiv_id}{version or ''}" if arxiv_id else "https://example.org/p"),
        arxiv_id=arxiv_id,
        version=version,
    )


def make_first_page_pdf(*, footnote: bool = False, abstract: bool = True, body_repeat: int = 10) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((150, 80), "Calibrated Verifiers for Agents", fontsize=16)
    page.insert_text((200, 110), "Alice Smith1, Bob Jones2", fontsize=11)
    if not footnote:
        page.insert_text((180, 128), "1 MIT CSAIL, 2 Stanford", fontsize=10)
    page.insert_text((180, 142), "alice@example.org", fontsize=9)
    if abstract:
        page.insert_text((280, 180), "Abstract", fontsize=12)
    page.insert_textbox(
        pymupdf.Rect(72, 190, 540, 700), "We calibrate the checker on held-out data. " * body_repeat, fontsize=10
    )
    if footnote:
        footer = "1 School of Informatics, University of Edinburgh. 2 Tsinghua University."
        page.insert_text((72, 740), footer, fontsize=8)
    return doc.tobytes()


# ---------------------------------------------------------------------------
# HTML author block
# ---------------------------------------------------------------------------


def test_html_block_from_captured_page_drops_emails():
    text = extract_html_author_block(_fixture("2610.30418v1"))
    assert text.startswith("Daniel C. Ruiz Luca Luceri & Emilio Ferrara")
    assert "Information Sciences Institute" in text and "University of Southern California" in text
    assert "@" not in text and "isi.edu" not in text and "Email" not in text
    assert "Abstract" not in text and "mosaic" not in text  # nothing after div.ltx_authors
    assert "\n" not in text and "  " not in text


def test_html_block_keeps_repeated_affiliation_once():
    text = extract_html_author_block(_fixture("2610.29755v1"))
    for name in ("Chao Jiang", "Ding Wang", "Yuwei Liu", "Lin Huang"):
        assert name in text
    assert text.count("Shenzhen University") == 1 and text.count("Ant Group") == 1
    assert "Shenzhen University, College of Computer Science" in text  # no " ," from tag joins
    assert "@" not in text and "chaoccc22" not in text and "orcid" not in text.lower()
    assert "PACMPL" not in text  # pubnotes sit outside div.ltx_authors


def test_html_block_from_author_list_section_when_front_matter_is_empty():
    # This page has no div.ltx_authors and nothing between the title and the abstract.
    text = extract_html_author_block(_fixture("2610.29932v1"))
    assert "Qiuyang Mang" in text and "UC Berkeley" in text and "University of Waterloo" in text
    assert "\\color" not in text  # LaTeXML ltx_ERROR macro names are dropped
    assert "†" in text and "\\dagger" not in text  # rendered math symbol, not its TeX source
    assert "Author list" not in text and "FrontierCS" not in text


SYNTHETIC_AUTHORS = """<html><body><article class="ltx_document">
<h1 class="ltx_title ltx_title_document">A Synthetic Paper</h1>
<div class="ltx_authors">
<span class="ltx_creator ltx_role_author"><span class="ltx_personname">Ada Lovelace<math alttext="{}^{1}"><semantics><msup><mi></mi><mn>1</mn></msup><annotation encoding="application/x-tex">{}^{1}</annotation></semantics></math></span>
<span class="ltx_author_notes"><span class="ltx_author_notes_content">
<span class="ltx_contact ltx_role_affiliation">MIT CSAIL<br class="ltx_break"/>Cambridge, MA</span>
<span class="ltx_contact ltx_role_email"><a href="mailto:ada@mit.edu">ada@mit.edu</a></span>
</span></span></span>
<span class="ltx_author_before">, </span>
<span class="ltx_creator ltx_role_author"><span class="ltx_personname">Alan Turing</span>
<span class="ltx_author_notes"><span class="ltx_author_notes_content">
<span class="ltx_contact ltx_role_affiliation">Google DeepMind</span>
<span class="ltx_contact ltx_role_address">London, UK</span>
<span class="ltx_contact ltx_role_email">turing@deepmind.com</span>
</span></span></span>
</div>
<div class="ltx_abstract"><h6 class="ltx_title ltx_title_abstract">Abstract</h6><p class="ltx_p">We study things.</p></div>
</article></body></html>"""


def test_html_block_synthetic_affiliation_spans():
    text = extract_html_author_block(SYNTHETIC_AUTHORS)
    assert text == "Ada Lovelace 1 MIT CSAIL Cambridge, MA, Alan Turing Google DeepMind London, UK"
    assert "@" not in text and "{}^" not in text


SYNTHETIC_NO_AUTHOR_DIV = """<html><body><article class="ltx_document">
<h1 class="ltx_title ltx_title_document">Title <span>With Span</span></h1>
<div class="ltx_authors">  </div>
<figure class="ltx_figure"><figcaption>Figure 1: Teaser of the method.</figcaption></figure>
<div class="ltx_para"><p class="ltx_p">Grace Hopper (Carnegie Mellon University), Edsger Dijkstra (contact: ed@utexas.edu)</p></div>
<div class="ltx_abstract"><h6 class="ltx_title ltx_title_abstract">Abstract</h6><p class="ltx_p">We study things.</p></div>
<section class="ltx_section"><h2>1 Introduction</h2><p>Intro text.</p></section>
</article></body></html>"""


def test_html_block_falls_back_to_title_abstract_region():
    text = extract_html_author_block(SYNTHETIC_NO_AUTHOR_DIV)
    assert text == "Grace Hopper (Carnegie Mellon University), Edsger Dijkstra (contact:)"
    assert "With Span" not in text and "Teaser" not in text and "We study" not in text


def test_html_block_empty_when_page_has_no_author_text():
    html = (
        '<html><body><article><h1 class="ltx_title_document">Only a title</h1>'
        '<div class="ltx_abstract"><p>Abstract text.</p></div>'
        '<section class="ltx_section"><h2>1 Introduction</h2><p>Body.</p></section></article></body></html>'
    )
    assert extract_html_author_block(html) == ""
    assert extract_html_author_block("<html><body><p>conversion failed</p></body></html>") == ""


def test_html_block_is_capped():
    names = "".join(
        f'<span class="ltx_creator ltx_role_author"><span class="ltx_personname">Author Number {i}</span>'
        f'<span class="ltx_contact ltx_role_affiliation">Institute {i} of Long Names</span></span>'
        for i in range(200)
    )
    html = f'<html><body><h1 class="ltx_title_document">T</h1><div class="ltx_authors">{names}</div></body></html>'
    text = extract_html_author_block(html)
    assert len(text) <= MAX_BLOCK_CHARS + 2 and text.endswith(" …")
    assert text.startswith("Author Number 0 Institute 0 of Long Names")


# ---------------------------------------------------------------------------
# PDF author block
# ---------------------------------------------------------------------------


def test_pdf_block_stops_at_abstract():
    text = extract_pdf_author_block(make_first_page_pdf())
    assert text == "Calibrated Verifiers for Agents Alice Smith1, Bob Jones2 1 MIT CSAIL, 2 Stanford"


def test_pdf_block_adds_footnote_affiliations():
    text = extract_pdf_author_block(make_first_page_pdf(footnote=True))
    assert text.startswith("Calibrated Verifiers for Agents Alice Smith1, Bob Jones2 … ")
    assert text.endswith("1 School of Informatics, University of Edinburgh. 2 Tsinghua University.")
    assert len(text) < 700


def test_pdf_block_without_abstract_takes_first_chars_and_bottom():
    text = extract_pdf_author_block(make_first_page_pdf(footnote=True, abstract=False, body_repeat=80))
    assert text.startswith("Calibrated Verifiers for Agents Alice Smith1, Bob Jones2 We calibrate")
    assert "University of Edinburgh" in text  # from the page bottom
    assert len(text) <= MAX_BLOCK_CHARS + 3 + affiliations.FOOTNOTE_CHARS + 2


def test_pdf_block_repairs_spacing_accents():
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((150, 80), "A Title", fontsize=16)
    page.insert_text((150, 110), "Mary Joe Medlej", fontsize=11)
    # Latin-1 spacing accents only: the built-in PDF font has no glyph for U+02C6.
    page.insert_text((150, 128), "Universit´e de Montr´eal, TU M¨unchen", fontsize=10)
    page.insert_text((150, 180), "Abstract", fontsize=12)
    assert extract_pdf_author_block(doc.tobytes()) == "A Title Mary Joe Medlej Université de Montréal, TU München"


def test_has_institution_words():
    assert has_institution_words("Dept. of CS, Stanford University")
    assert has_institution_words("Google DeepMind")
    assert not has_institution_words("Alice Smith, Bob Jones")
    assert not has_institution_words("")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_classify_tiers_and_drops_non_list_matches():
    blocks = {
        "u0": ("Ada (MIT CSAIL), Alan (DeepMind)", "html"),
        "u1": ("Grace (University of Edinburgh)", "pdf"),
        "u2": ("Edsger (UT Austin)", "html"),
        "u3": ("Barbara (Massachusetts Institute of Technology)", "html"),
    }
    client = _client(
        [
            _results(
                (0, ["MIT", "Google DeepMind", "mit"], ["Google DeepMind", "MIT", "MIT"]),
                (1, ["University of Edinburgh"], ["University of Edinburgh", "Edinburgh Uni"]),
                (2, ["UT Austin"], []),
                (3, ["MIT"], ["Massachusetts Institute of Technology", " MIT "]),
            )
        ]
    )

    result = classify_affiliations(client, "m", blocks, TRUSTED)

    assert list(result) == ["u0", "u1", "u2", "u3"]
    assert result["u0"] == Affiliations("u0", ["MIT", "Google DeepMind"], ["MIT", "Google DeepMind"], 1, "html")
    assert result["u1"] == Affiliations("u1", ["University of Edinburgh"], ["University of Edinburgh"], 2, "pdf")
    assert result["u2"] == Affiliations("u2", ["UT Austin"], [], 0, "html")
    assert result["u3"].matched == ["MIT"] and result["u3"].tier == 1  # non-list string dropped

    call = client.chat.completions.create.call_args_list[0].kwargs
    assert call["model"] == "m" and call["response_format"] == {"type": "json_object"}
    system, user = _messages(client, 0)[0]["content"], _messages(client, 0)[1]["content"]
    assert "1. MIT\n2. Stanford University\n3. Carnegie Mellon University\n4. Google DeepMind" in system
    assert "5. University of Edinburgh\n6. Tsinghua University" in system
    assert "0. Author/affiliation text: Ada (MIT CSAIL), Alan (DeepMind)" in user
    assert "3. Author/affiliation text: Barbara" in user


def test_classify_tier1_wins_over_tier2():
    client = _client(
        [_results((0, ["Tsinghua University", "Stanford"], ["Tsinghua University", "Stanford University"]))]
    )
    result = classify_affiliations(client, "m", {"u": ("text", "html")}, TRUSTED)
    assert result["u"].tier == 1 and result["u"].matched == ["Stanford University", "Tsinghua University"]


def test_classify_bad_json_twice_gives_tier_zero():
    client = _client(["not json", "still not json"])
    result = classify_affiliations(client, "m", {"u": ("Ada (MIT)", "pdf")}, TRUSTED)
    assert result["u"] == Affiliations("u", [], [], 0, "pdf")
    assert client.chat.completions.create.call_count == 2
    assert _messages(client, 1)[1]["content"].startswith(JSON_RETRY_PREFIX)


def test_classify_llm_exception_degrades():
    client = _client([RuntimeError("boom"), RuntimeError("boom again")])
    result = classify_affiliations(client, "m", {"u": ("Ada (MIT)", "html")}, TRUSTED)
    assert result["u"].tier == 0 and result["u"].source == "html"


def test_classify_out_of_range_index_invalidates_response_then_retry():
    client = _client(
        [
            _results((0, ["MIT"], ["MIT"]), (5, ["X"], [])),
            _results((0, ["MIT"], ["MIT"]), (1, ["Tsinghua University"], ["Tsinghua University"])),
        ]
    )
    result = classify_affiliations(client, "m", {"a": ("t a", "html"), "b": ("t b", "html")}, TRUSTED)
    assert client.chat.completions.create.call_count == 2
    assert result["a"].tier == 1 and result["b"].tier == 2


def test_classify_partial_response_retries_then_degrades_missing():
    client = _client([_results((0, ["MIT"], ["MIT"])), _results((0, ["Wrong"], []))])
    result = classify_affiliations(client, "m", {"a": ("t a", "html"), "b": ("t b", "pdf")}, TRUSTED)
    assert client.chat.completions.create.call_count == 2
    assert result["a"].matched == ["MIT"]  # first answer kept
    assert result["b"] == Affiliations("b", [], [], 0, "pdf")


def test_classify_batches_and_skips_empty_text():
    blocks = {f"u{i}": (f"text {i}", "html") for i in range(23)}
    blocks["empty"] = ("", "none")
    client = _client(
        [
            _results(*[(i, [], []) for i in range(10)]),
            _results(*[(i, ["MIT"] if i == 0 else [], ["MIT"] if i == 0 else []) for i in range(10)]),
            _results(*[(i, [], []) for i in range(3)]),
        ]
    )

    result = classify_affiliations(client, "m", blocks, TRUSTED)

    assert client.chat.completions.create.call_count == 3
    assert set(result) == set(blocks)
    assert result["u10"].tier == 1 and result["u0"].tier == 0
    assert result["empty"] == Affiliations("empty", [], [], 0, "none")
    second_user = _messages(client, 1)[1]["content"]
    assert "0. Author/affiliation text: text 10" in second_user and "text 9\n" not in second_user
    assert all("(empty)" not in _messages(client, i)[1]["content"] for i in range(3))


def test_classify_drops_list_entries_the_paper_does_not_name():
    # Seen live: the model copied the whole trusted list into "matched" for one paper.
    text = "Qiuyang Mang UC Berkeley Wenhao Chai Princeton University Qu Ao MIT Sherry Ruan Stanford University"
    institutions = ["UC Berkeley", "Princeton University", "MIT", "Stanford University"]
    everything = TRUSTED["tier1"] + TRUSTED["tier2"]
    client = _client([_results((0, institutions, everything), (1, ["UC Berkeley"], everything))])

    result = classify_affiliations(client, "m", {"a": (text, "html"), "b": ("Ann (UC Berkeley)", "html")}, TRUSTED)

    assert result["a"].matched == ["MIT", "Stanford University"] and result["a"].tier == 1
    assert result["b"].matched == [] and result["b"].tier == 0


@pytest.mark.parametrize(
    "text, institutions, entry",
    [
        ("Ada (Massachusetts Institute of Technology)", ["Massachusetts Institute of Technology"], "MIT"),
        ("Ada (MIT CSAIL)", [], "MIT"),
        ("Alan (DeepMind, London)", ["DeepMind"], "Google DeepMind"),
        ("Grace (CMU)", ["CMU"], "Carnegie Mellon University"),
        ("Edsger (Stanford AI Lab)", ["Stanford"], "Stanford University"),
        ("Kurt (ETH Zürich)", ["ETH Zurich"], "ETH Zürich"),
        ("Yann (Facebook AI Research)", ["Meta"], "Meta"),  # the LLM wrote the list's spelling
    ],
)
def test_aliases_and_sub_units_count_as_supported(text, institutions, entry):
    client = _client([_results((0, institutions, [entry]))])
    trusted = {"tier1": [entry], "tier2": []}
    result = classify_affiliations(client, "m", {"u": (text, "pdf")}, trusted)
    assert result["u"].matched == [entry] and result["u"].tier == 1


def test_classify_with_empty_trusted_lists():
    client = _client([_results((0, ["MIT"], ["MIT"]))])
    result = classify_affiliations(client, "m", {"u": ("Ada (MIT)", "html")}, {"tier1": [], "tier2": []})
    assert result["u"] == Affiliations("u", ["MIT"], [], 0, "html")
    assert "(none)" in _messages(client, 0)[0]["content"]


# ---------------------------------------------------------------------------
# Fetching + end to end
# ---------------------------------------------------------------------------


def test_fetch_author_blocks_sources_and_failures():
    html_entry = _entry("2610.30418")
    pdf_entry = _entry("2610.11111")
    hf_entry = _entry("2610.22222", version=None, url="https://huggingface.co/papers/2610.22222")
    broken = _entry("2610.33333")
    no_id = _entry(None, version=None)
    names_only = _entry("2610.44444")
    session = FakeSession(
        {
            "https://arxiv.org/html/2610.30418v1": FakeResponse(200, _fixture("2610.30418v1").encode()),
            "https://arxiv.org/pdf/2610.11111v1": FakeResponse(200, make_first_page_pdf()),
            "https://arxiv.org/html/2610.22222v3": FakeResponse(200, SYNTHETIC_AUTHORS.encode()),
            "https://arxiv.org/html/2610.33333v1": requests.ConnectionError("reset"),
            "https://arxiv.org/pdf/2610.33333v1": FakeResponse(200, b"<html>not a pdf</html>"),
            "https://arxiv.org/html/2610.44444v1": FakeResponse(
                200,
                b'<html><body><div class="ltx_authors"><span class="ltx_personname">Ann Bee, Cy Dee'
                b"</span></div></body></html>",
            ),
            "https://arxiv.org/pdf/2610.44444v1": FakeResponse(200, make_first_page_pdf()),
        }
    )
    details = {hf_entry.url: PaperDetails("2610.22222", "v3", "T", "A", [])}

    blocks = fetch_author_blocks(
        [html_entry, pdf_entry, hf_entry, broken, no_id, names_only, html_entry], details, session=session
    )

    assert list(blocks) == [html_entry.url, pdf_entry.url, hf_entry.url, broken.url, no_id.url, names_only.url]
    assert blocks[html_entry.url][1] == "html" and "University of Southern California" in blocks[html_entry.url][0]
    assert blocks[pdf_entry.url] == (
        "Calibrated Verifiers for Agents Alice Smith1, Bob Jones2 1 MIT CSAIL, 2 Stanford",
        "pdf",
    )
    assert blocks[hf_entry.url][1] == "html" and "Google DeepMind" in blocks[hf_entry.url][0]  # version from details
    assert blocks[broken.url] == ("", "none")
    assert blocks[no_id.url] == ("", "none")
    assert blocks[names_only.url][1] == "pdf"  # HTML had names but no institution, the PDF has one
    assert session.calls.count("https://arxiv.org/html/2610.30418v1") == 1  # duplicate entry fetched once
    assert "https://arxiv.org/pdf/2610.30418v1" not in session.calls


def test_html_names_only_kept_when_pdf_has_nothing_better():
    entry = _entry("2610.55555")
    session = FakeSession(
        {
            "https://arxiv.org/html/2610.55555v1": FakeResponse(
                200, b'<html><body><div class="ltx_authors">Ann Bee, Cy Dee</div></body></html>'
            ),
        }
    )
    assert fetch_author_blocks([entry], {}, session=session) == {entry.url: ("Ann Bee, Cy Dee", "html")}


def test_lookup_affiliations_end_to_end():
    entries = [_entry("2610.30418"), _entry("2610.11111"), _entry(None, version=None), _entry("2610.33333")]
    session = FakeSession(
        {
            "https://arxiv.org/html/2610.30418v1": FakeResponse(200, _fixture("2610.30418v1").encode()),
            "https://arxiv.org/pdf/2610.11111v1": FakeResponse(200, make_first_page_pdf()),
            "https://arxiv.org/html/2610.33333v1": FakeResponse(500, b"oops"),
            "https://arxiv.org/pdf/2610.33333v1": FakeResponse(503, b"busy"),
        }
    )
    client = _client(
        [
            _results(
                (0, ["Information Sciences Institute", "University of Southern California"], ["USC"]),
                (1, ["MIT", "Stanford University"], ["MIT", "Stanford University"]),
            )
        ]
    )

    result = lookup_affiliations(client, "m", entries, {}, TRUSTED, session=session)

    assert set(result) == {e.url for e in entries}
    a0, a1, a2, a3 = (result[e.url] for e in entries)
    assert (a0.source, a0.tier, a0.matched) == ("html", 0, [])  # "USC" is not a list string
    assert a0.institutions == ["Information Sciences Institute", "University of Southern California"]
    assert (a1.source, a1.tier, a1.matched) == ("pdf", 1, ["MIT", "Stanford University"])
    assert a2 == Affiliations(entries[2].url, [], [], 0, "none")
    assert a3 == Affiliations(entries[3].url, [], [], 0, "none")
    assert client.chat.completions.create.call_count == 1
    user = _messages(client, 0)[1]["content"]
    assert "University of Southern California" in user and "1 MIT CSAIL, 2 Stanford" in user
    assert "2. Author" not in user  # papers without text are never sent
