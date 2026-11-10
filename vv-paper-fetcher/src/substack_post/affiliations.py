"""Find each paper's author institutions and match them to a trusted list.

The post keeps at most a fixed number of papers and prefers papers from
trusted institutions, so this step gives every digest entry an Affiliations
record with a tier: 1 = matches a tier-1 entry, 2 = matches only tier-2
entries, 0 = no match or unknown.

Per paper, the author/affiliation text comes from the first source that works:

1. arXiv HTML (https://arxiv.org/html/<id>vN): the text of LaTeXML's
   ``div.ltx_authors`` (emails and ORCID links dropped, a repeated affiliation
   kept once); when that is missing or empty, the text between the title
   (``h1.ltx_title_document``) and the abstract (``div.ltx_abstract``); then a
   section headed "Authors" / "Author list" / "Affiliations" / "Contributors".
2. PDF (no HTML, no author text in it, or author text without a single
   institution-like word): page 1 of https://arxiv.org/pdf/<id>vN from the top
   down to the line that starts with "Abstract" (else its first 2,000
   characters), plus the page's last 600 characters when the top part has no
   institution-like word, because some templates put affiliations in footnotes.

Text is whitespace-collapsed and capped at about 2,000 characters.

Then batched LLM calls (OpenRouter, JSON mode, one retry with the strict
"respond with ONLY the JSON" reminder, like writer.py) name each paper's
institutions and copy the matching entries of the trusted lists. Python keeps
only matches that are EXACT list strings and that the paper's own text or the
LLM's institution names support (a loose word/acronym check that catches the
LLM copying list entries no author belongs to), then computes the tier itself.

One paper's failure never stops the others: it gets source "none" or tier 0.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pymupdf
import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString, PreformattedString, Tag

from . import prompts
from .figures import HTML_BASE, PDF_BASE, _fetch_throttle
from .models import DigestEntry, PaperDetails
from .paper_details import USER_AGENT, new_session
from .writer import _as_int, _call_llm, _load_json

logger = logging.getLogger(__name__)

TIMEOUT_S = 60
MAX_BLOCK_CHARS = 2000  # cap on the author text taken from the HTML or the top of PDF page 1
FOOTNOTE_CHARS = 600  # bottom of PDF page 1, added when the top names no institution
BATCH_SIZE = 10  # papers per LLM call

# A word that suggests the text names an organization. Used only to decide
# whether to look further (PDF fallback, PDF footnotes), never to classify.
_INSTITUTION_RE = re.compile(
    r"univ|institut|college|school|academy|laborator|\blabs?\b|research|\bcent(?:er|re)\b|"
    r"department|\bdept\b|faculty|polytechn|hochschule|[eé]cole|\binc\b|\bcorp|\bltd\b|\bllc\b|"
    r"gmbh|company|hospital|foundation|google|deepmind|microsoft|openai|anthropic|\bmeta\b|"
    r"nvidia|\bibm\b|amazon|apple|alibaba|tencent|baidu|bytedance|huawei|\bmit\b|\bcmu\b|"
    r"\beth\b|\bepfl\b|\bucla\b|\bucl\b|\bkaist\b|inria|cnrs|max planck",
    re.IGNORECASE,
)
# "ada@mit.edu" and grouped "{ada,alan}@mit.edu" addresses.
_EMAIL_RE = re.compile(r"(?:\{[^{}@]*\}|[\w.+-]+)\s*@\s*[\w-]+(?:\.[\w-]+)+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:)\]])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[])\s+")
_ABSTRACT_LINE_RE = re.compile(r"^abstract\b", re.IGNORECASE)
_ARXIV_STAMP_RE = re.compile(r"^arXiv:\d{4}\.\d{4,5}(?:v\d+)?\b")
_AUTHOR_SECTION_RE = re.compile(
    r"^(?:[A-Z]?\d+(?:\.\d+)*\.?\s+)?(?:authors?(?:\s+list)?|authors?\s+and\s+affiliations?|"
    r"affiliations?|contributors?|list\s+of\s+authors)$",
    re.IGNORECASE,
)

# Removed before any text is taken: figure/table floats (a teaser figure can sit
# between the title and the abstract), emails, ORCID icons, LaTeXML's
# undefined-macro markers (e.g. a literal "\color"); _prepare() also drops the
# TeX source that MathML carries along.
_DROP_SELECTORS = ("script", "style", "svg", "figure", ".ltx_role_email", ".ltx_role_orcid", ".ltx_ERROR")
_STOP_TAGS = {"section", "h2", "nav", "footer"}

# LaTeX PDFs often extract accents as a separate spacing mark before the letter
# ("Universit´e Cˆote"); these become the combining mark after it.
_SPACING_ACCENTS = {"´": "́", "ˆ": "̂", "¨": "̈", "˜": "̃", "¸": "̧"}
_SPACING_ACCENT_RE = re.compile("([" + "".join(_SPACING_ACCENTS) + r"])\s?([A-Za-z])")

# Words too common in organization names to show that a trusted entry is named.
_ACRONYM_SKIP = frozenset("a an and at de del der des di du for in la le of the".split())
_GENERIC_WORDS = _ACRONYM_SKIP | frozenset(
    "university universite universitat universidad universita univ institute institut instituto "
    "technology technological research lab labs laboratory laboratories college school department "
    "dept center centre national state inc ltd llc corp corporation company group ai".split()
)


@dataclass
class Affiliations:
    """A paper's author institutions and how they match the trusted-institutions list."""

    url: str  # paper URL (key), as in DigestEntry.url
    institutions: List[str]  # de-duplicated, short display form, e.g. "MIT", "Google DeepMind"
    matched: List[str]  # trusted-list entries that match, as EXACT strings from the config lists
    tier: int  # 1 = matched a tier-1 entry, 2 = matched only tier-2, 0 = no match or unknown
    source: str  # "html" | "pdf" | "none"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    """Collapse whitespace, drop email addresses, and tidy spaces around punctuation."""
    text = _EMAIL_RE.sub(" ", text or "")
    text = " ".join(text.split())
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return _SPACE_AFTER_OPEN_RE.sub(r"\1", text)


def _cap(text: str, limit: int = MAX_BLOCK_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.8:
        cut = cut[:space]
    return cut.rstrip() + " …"


def has_institution_words(text: str) -> bool:
    """True when the text contains a word that usually belongs to an organization name."""
    return bool(text) and _INSTITUTION_RE.search(text) is not None


def _fix_spacing_accents(text: str) -> str:
    fixed = _SPACING_ACCENT_RE.sub(lambda m: m.group(2) + _SPACING_ACCENTS[m.group(1)], text)
    return unicodedata.normalize("NFC", fixed)


# ---------------------------------------------------------------------------
# HTML author block
# ---------------------------------------------------------------------------


def _prepare(soup: BeautifulSoup) -> None:
    for selector in _DROP_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()
    for tag in soup.find_all(["annotation", "annotation-xml"]):
        tag.decompose()  # keeps the rendered symbol ("†", "1") of math superscripts


def _authors_div_text(soup: BeautifulSoup) -> str:
    authors = soup.select_one("div.ltx_authors")
    if authors is None:
        return ""
    seen = set()
    for contact in authors.select(".ltx_role_affiliation, .ltx_role_address"):
        key = _clean(contact.get_text(" ")).lower()
        if key in seen:
            contact.decompose()  # many templates repeat each author's affiliation
        else:
            seen.add(key)
    return _clean(authors.get_text(" "))


def _title_to_abstract_text(soup: BeautifulSoup) -> str:
    title = soup.select_one("h1.ltx_title_document")
    if title is None:
        return ""
    inside_title = {id(node) for node in title.descendants}
    parts: List[str] = []
    for node in title.next_elements:
        if isinstance(node, Tag):
            if node.name in _STOP_TAGS or "ltx_abstract" in (node.get("class") or []):
                break
            continue
        if id(node) in inside_title:
            continue
        if isinstance(node, NavigableString) and not isinstance(node, PreformattedString):
            parts.append(str(node))
    return _clean(" ".join(parts))


def _author_section_text(soup: BeautifulSoup) -> str:
    for section in soup.find_all("section"):
        heading = section.find(["h2", "h3", "h4", "h5", "h6"])
        if heading is None or not _AUTHOR_SECTION_RE.match(_clean(heading.get_text(" "))):
            continue
        heading.extract()
        text = _clean(section.get_text(" "))
        if text:
            return text
    return ""


def extract_html_author_block(html: str) -> str:
    """Author/affiliation text of an arXiv HTML page, capped at MAX_BLOCK_CHARS; "" if none."""
    soup = BeautifulSoup(html, "html.parser")
    _prepare(soup)
    for step in (_authors_div_text, _title_to_abstract_text, _author_section_text):
        text = step(soup)
        if text:
            return _cap(text)
    return ""


# ---------------------------------------------------------------------------
# PDF author block
# ---------------------------------------------------------------------------


def extract_pdf_author_block(pdf_bytes: bytes) -> str:
    """Author/affiliation text from page 1 of a PDF; "" when the page has no text."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        if doc.page_count == 0:
            return ""
        raw = doc[0].get_text("text", sort=True)
    finally:
        doc.close()

    lines = [" ".join(line.split()) for line in _fix_spacing_accents(raw).splitlines()]
    lines = [line for line in lines if line and not _ARXIV_STAMP_RE.match(line)]
    cut = next((i for i, line in enumerate(lines) if _ABSTRACT_LINE_RE.match(line)), None)
    top_full = _clean(" ".join(lines[:cut] if cut is not None else lines))
    top = _cap(top_full)
    if has_institution_words(top):
        return top

    # Footnote affiliations: the bottom of the page, unless `top` already holds the whole page.
    if cut is not None:
        rest = _clean(" ".join(lines[cut:]))
    else:
        rest = top_full if len(top_full) > MAX_BLOCK_CHARS else ""
    tail = rest[-FOOTNOTE_CHARS:].strip()
    if tail:
        top = f"{top} … {tail}" if top else tail
    return top


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _get(session: requests.Session, url: str) -> requests.Response:
    _fetch_throttle.wait()  # shared with figures.py: >= 1 s between arxiv.org requests
    return session.get(url, timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT})


def _version(entry: DigestEntry, details: Optional[PaperDetails]) -> str:
    return entry.version or (details.version if details else None) or ""


def _html_block(arxiv_id: str, version: str, session: requests.Session) -> str:
    resp = _get(session, f"{HTML_BASE}{arxiv_id}{version}")
    if resp.status_code == 404:
        logger.info("No arXiv HTML for %s%s (404); will use the PDF", arxiv_id, version)
        return ""
    resp.raise_for_status()
    return extract_html_author_block(resp.text)


def _pdf_block(arxiv_id: str, version: str, session: requests.Session) -> str:
    url = f"{PDF_BASE}{arxiv_id}{version}"
    resp = _get(session, url)
    resp.raise_for_status()
    if not resp.content.startswith(b"%PDF"):
        raise ValueError(f"{url} did not return a PDF")
    return extract_pdf_author_block(resp.content)


def _author_block(entry: DigestEntry, details: Optional[PaperDetails], session: requests.Session) -> Tuple[str, str]:
    if not entry.arxiv_id:
        return "", "none"
    version = _version(entry, details)

    html_text = ""
    try:
        html_text = _html_block(entry.arxiv_id, version, session)
    except Exception as exc:  # noqa: BLE001 - the PDF is still an option
        logger.warning("arXiv HTML fetch failed for %s: %s", entry.arxiv_id, exc)
    if has_institution_words(html_text):
        return html_text, "html"

    pdf_text = ""
    try:
        pdf_text = _pdf_block(entry.arxiv_id, version, session)
    except Exception as exc:  # noqa: BLE001 - one paper must never kill the run
        logger.warning("arXiv PDF fetch failed for %s: %s", entry.arxiv_id, exc)
    if pdf_text and (not html_text or has_institution_words(pdf_text)):
        return pdf_text, "pdf"
    if html_text:
        return html_text, "html"
    return "", "none"


def fetch_author_blocks(
    entries: List[DigestEntry],
    details: Mapping[str, PaperDetails],
    session: Optional[requests.Session] = None,
) -> Dict[str, Tuple[str, str]]:
    """Return {entry.url: (author/affiliation text, source)} with source "html" | "pdf" | "none".

    Entries without an arXiv id, or whose HTML and PDF both fail, get ("", "none").
    Never raises for one paper.
    """
    session = session or new_session()
    details = details or {}
    blocks: Dict[str, Tuple[str, str]] = {}
    for entry in entries:
        if entry.url in blocks:
            continue
        try:
            block = _author_block(entry, prompts.details_for(entry, details), session)
        except Exception as exc:  # noqa: BLE001 - one paper must never kill the run
            logger.warning("Author lookup failed for %s: %s", entry.url, exc)
            block = ("", "none")
        blocks[entry.url] = block
        logger.info("Author text for %s from %s (%d chars)", entry.arxiv_id or entry.url, block[1], len(block[0]))
    found = sum(1 for _, source in blocks.values() if source != "none")
    logger.info("Found author text for %d/%d papers", found, len(blocks))
    return blocks


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You read the author and affiliation text of research papers. For each paper you report (1) the institutions its authors belong to and (2) which entries of a fixed list of trusted institutions those institutions are.

Rules:
- Use ONLY the author/affiliation text given for the paper. Do not guess an institution that is not named in that text: not from author names, not from email domains, not from your own knowledge of the authors. When the text is empty or names no institution, return empty lists for that paper.
- "institutions": each distinct organization (university, company, research lab, institute) the authors are affiliated with, once, in its short display form, without departments, sub-labs, or addresses. Examples: "MIT" for "MIT CSAIL, Cambridge, MA"; "Google DeepMind" for "DeepMind, London, UK"; "Stanford University" for "Department of Computer Science, Stanford University". When an institution is one of the trusted-list entries below, write it exactly as the list writes it.
- "matched": the trusted-list entries that one of the paper's institutions IS, each copied EXACTLY, character for character, from the lists below (never the paper's spelling, never a list number). Match by meaning: a full name and its abbreviation are the same institution ("Massachusetts Institute of Technology" = "MIT"), so are old and new names ("DeepMind" = "Google DeepMind"), and a sub-unit counts as its institution ("MIT CSAIL" counts as "MIT"). Different organizations with similar names do not match (for example, "University of Washington" is not "Washington University in St. Louis"). Every "matched" entry must be one of this paper's "institutions"; most papers match no entry or only a few, so never copy the list itself. When nothing matches, "matched" is [].
- Respond only with the specified JSON, no other text."""


def _numbered(names: Sequence[str], start: int) -> str:
    if not names:
        return "(none)"
    return "\n".join(f"{start + i}. {name}" for i, name in enumerate(names))


def build_system_prompt(tier1: Sequence[str], tier2: Sequence[str]) -> str:
    """The rules plus both trusted lists, numbered continuously, with their exact strings."""
    return (
        SYSTEM_PROMPT
        + "\n\nTrusted institutions, tier 1:\n"
        + _numbered(tier1, 1)
        + "\n\nTrusted institutions, tier 2:\n"
        + _numbered(tier2, len(tier1) + 1)
    )


def build_user_message(texts: Sequence[str]) -> str:
    items = [f"{i}. Author/affiliation text: {text or '(empty)'}" for i, text in enumerate(texts)]
    return (
        "Papers:\n\n"
        + "\n\n".join(items)
        + '\n\nRespond with ONLY this JSON shape: {"results": [{"index": <int>, '
        '"institutions": ["<short institution name>", ...], '
        '"matched": ["<exact trusted-list string>", ...]}, ...]} '
        "— one entry per paper, indices matching the list above."
    )


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]


def _parse_results(data: Any, batch_len: int) -> Dict[int, Tuple[List[str], List[str]]]:
    """Validate a classification response; return {batch index: (institutions, matched)}.

    An index outside the batch makes the whole response invalid, as in the
    KEEP/DROP filter; a paper missing from the response simply has no entry.
    """
    results = data["results"]
    if not isinstance(results, list):
        raise ValueError("'results' is not a list")
    by_index: Dict[int, Tuple[List[str], List[str]]] = {}
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("'results' items must be objects")
        index = _as_int(item.get("index"))
        if index is None or not 0 <= index < batch_len:
            raise ValueError(f"index {item.get('index')!r} is outside the batch")
        by_index.setdefault(index, (_string_list(item.get("institutions")), _string_list(item.get("matched"))))
    return by_index


def _classify_batch(
    client: Any, model: str, system: str, texts: List[str]
) -> Tuple[Dict[int, Tuple[List[str], List[str]]], str]:
    """Run one batch with one retry; return (results by index, reason if incomplete)."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_message(texts)},
    ]
    best: Dict[int, Tuple[List[str], List[str]]] = {}
    reason = ""
    for attempt in range(2):
        attempt_messages = messages if attempt == 0 else prompts.with_json_retry_prefix(messages)
        try:
            raw = _call_llm(client, model, attempt_messages)
        except Exception as exc:  # noqa: BLE001 - one batch must never kill the run
            reason = f"LLM call failed: {exc}"
            logger.warning("Affiliation LLM call failed (attempt %d): %s", attempt + 1, exc)
            continue
        try:
            by_index = _parse_results(_load_json(raw), len(texts))
        except (ValueError, KeyError, TypeError) as exc:
            reason = f"invalid JSON response: {exc}"
            logger.warning("Affiliation batch: %s (attempt %d)", reason, attempt + 1)
            continue
        for index, result in by_index.items():
            best.setdefault(index, result)
        if len(best) == len(texts):
            return best, ""
        missing = sorted(set(range(len(texts))) - set(best))
        reason = f"no result returned for batch indices {missing}"
        logger.warning("Affiliation batch: %s (attempt %d)", reason, attempt + 1)
    return best, reason


def _trusted_list(trusted: Optional[Mapping[str, List[str]]], key: str) -> List[str]:
    values = (trusted or {}).get(key) or []
    return [v for v in values if isinstance(v, str) and v.strip()]


def _dedupe(names: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for name in names:
        clean = " ".join(name.split())
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            out.append(clean)
    return out


def _normalize(text: str) -> str:
    """Lowercase ASCII words: accents stripped, punctuation turned into spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text.lower()).split())


def _acronym(normalized: str) -> str:
    """Initials of the non-filler words ("carnegie mellon university" -> "cmu"); "" for one word."""
    words = [w for w in normalized.split() if w not in _ACRONYM_SKIP]
    return "".join(w[0] for w in words) if len(words) >= 2 else ""


def _is_supported(entry: str, institutions: List[str], text: str) -> bool:
    """Loose check that a trusted-list entry is named in the paper's text or the LLM's institutions.

    The LLM still decides what matches; this only rejects an entry that shares
    no distinctive word or acronym with anything the paper says (the LLM
    copying list entries no author belongs to). "MIT" is supported by "MIT
    CSAIL" or by an institution "Massachusetts Institute of Technology",
    "Google DeepMind" by "DeepMind", "Carnegie Mellon University" by "CMU".
    """
    entry_norm = _normalize(entry)
    institution_norms = [_normalize(name) for name in institutions]
    if entry_norm in institution_norms:
        return True
    evidence = set(_normalize(text).split())
    for norm in institution_norms:
        evidence.update(norm.split())
        evidence.add(_acronym(norm))
    evidence.discard("")
    words = entry_norm.split()
    distinctive = [w for w in words if w not in _GENERIC_WORDS] or words
    if any(w in evidence for w in distinctive):
        return True
    acronym = _acronym(entry_norm)
    return bool(acronym) and acronym in evidence


def _build(
    url: str,
    source: str,
    text: str,
    institutions: List[str],
    matched: List[str],
    tier1: List[str],
    tier2: List[str],
) -> Affiliations:
    """Keep only exact, supported trusted-list strings (in list order) and compute the tier."""
    order: Dict[str, int] = {}
    for name in tier1 + tier2:
        order.setdefault(name, len(order))
    kept: List[str] = []
    for name in matched:
        exact = name if name in order else name.strip()
        if exact not in order:
            logger.warning("Dropping match %r for %s: not an exact trusted-list entry", name, url)
            continue
        if not _is_supported(exact, institutions, text):
            logger.warning("Dropping match %r for %s: not named in the paper's affiliations", exact, url)
            continue
        if exact not in kept:
            kept.append(exact)
    kept.sort(key=order.__getitem__)

    tier1_set, tier2_set = set(tier1), set(tier2)
    if any(name in tier1_set for name in kept):
        tier = 1
    elif any(name in tier2_set for name in kept):
        tier = 2
    else:
        tier = 0
    return Affiliations(url=url, institutions=_dedupe(institutions), matched=kept, tier=tier, source=source)


def classify_affiliations(
    client: Any,
    model: str,
    author_blocks: Mapping[str, Tuple[str, str]],
    trusted: Mapping[str, List[str]],
    batch_size: int = BATCH_SIZE,
) -> Dict[str, Affiliations]:
    """Return one Affiliations per URL in ``author_blocks``.

    ``trusted`` is {"tier1": [...], "tier2": [...]} from the config. Papers with
    no author text are not sent to the LLM; papers whose batch fails twice get
    institutions=[], matched=[], tier=0 and keep their source.
    """
    tier1 = _trusted_list(trusted, "tier1")
    tier2 = _trusted_list(trusted, "tier2")
    system = build_system_prompt(tier1, tier2)

    results: Dict[str, Affiliations] = {}
    to_classify: List[Tuple[str, str, str]] = []  # (url, text, source)
    for url, (text, source) in author_blocks.items():
        text = (text or "").strip()
        if text:
            to_classify.append((url, text, source))
        else:
            results[url] = Affiliations(url=url, institutions=[], matched=[], tier=0, source=source)

    for start in range(0, len(to_classify), batch_size):
        batch = to_classify[start : start + batch_size]
        by_index, reason = _classify_batch(client, model, system, [text for _, text, _ in batch])
        if reason:
            failed = [url for i, (url, _, _) in enumerate(batch) if i not in by_index]
            logger.warning("Affiliation lookup failed for %d paper(s) (%s): %s", len(failed), reason, failed)
        for i, (url, text, source) in enumerate(batch):
            institutions, matched = by_index.get(i, ([], []))
            results[url] = _build(url, source, text, institutions, matched, tier1, tier2)

    ordered = {url: results[url] for url in author_blocks}
    counts = {tier: sum(1 for a in ordered.values() if a.tier == tier) for tier in (1, 2, 0)}
    logger.info(
        "Affiliations: %d tier-1, %d tier-2, %d untiered out of %d papers",
        counts[1], counts[2], counts[0], len(ordered),
    )
    return ordered


def lookup_affiliations(
    client: Any,
    model: str,
    entries: List[DigestEntry],
    details: Mapping[str, PaperDetails],
    trusted: Mapping[str, List[str]],
    session: Optional[requests.Session] = None,
) -> Dict[str, Affiliations]:
    """fetch_author_blocks + classify_affiliations; one Affiliations per entry.url."""
    blocks = fetch_author_blocks(entries, details, session=session)
    return classify_affiliations(client, model, blocks, trusted)
