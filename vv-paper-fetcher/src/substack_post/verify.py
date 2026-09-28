"""Deterministic checks from WRITING_SUBSTACK_POST.md sections 4, 5 and 7.

Hard errors send the draft back to the LLM for a rewrite; warnings only go
into the review email. The checks run on the rendered markdown (built by
render.py) and on the LLM-written parts of the post (title, intro, bodies),
so a link the LLM smuggles into a body is caught even though the per-paper
links themselves are exact by construction.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

from .models import DigestEntry, PaperDetails, PostDraft, VerifyResult
from .prompts import details_for
from .render import render_markdown

# Section 7: `grep -oE 'arxiv\.org/abs/[0-9]+\.[0-9]+v[0-9]+'`
ARXIV_ABS_RE = re.compile(r"arxiv\.org/abs/[0-9]+\.[0-9]+v[0-9]+")
# Any markdown link or image target: "[text](target)". Matching on "](" copes
# with brackets inside the link text.
LINK_TARGET_RE = re.compile(r"\]\(\s*<?([^)\s>]*)>?[^)]*\)")
BARE_URL_RE = re.compile(r"https?://\S+|\bwww\.\S+|\b(?:arxiv\.org|huggingface\.co)/\S*", re.IGNORECASE)
CONTRIBUTION_RE = re.compile(
    r"\b(verification|validation|validation-methodology|uncertainty[- ]quantification|UQ)\b"
    r"[^.]{0,60}\bcontribution\b",
    re.IGNORECASE,
)
ET_AL_RE = re.compile(r"\bet\.?\s+al\b", re.IGNORECASE)
MAIL_MERGE_RE = re.compile(r"\band\s+other\s+authors\b", re.IGNORECASE)
IMAGE_LINE_RE = re.compile(r"^!\[.*\]\(.*\)$", re.DOTALL)
PARAGRAPH_SPLIT_RE = re.compile(r"\n[ \t]*\n")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?][\"'”’)])\s+|(?<=[.!?])\s+")
TRAILING_CLOSERS_RE = re.compile(r"[\s\"'”’)\]*_]+$")

BANNED_PHRASES = [
    "in today's rapidly evolving",
    "rapidly evolving landscape",
    "game-changer",
    "game changer",
    "cutting-edge",
    "cutting edge",
    "revolutionary",
    "here's why this matters",
    "in conclusion",
    "let's dive in",
    "delve",
]

NUMBER_WORDS: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30,
}
_TENS = {"twenty": 20, "thirty": 30}
_UNITS = {w: n for w, n in NUMBER_WORDS.items() if n < 10}
_NAME_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv"}

MIN_BODY_WORDS = 80
MAX_BODY_WORDS = 200
MAX_SENTENCE_WORDS = 25
MIN_SURNAME_CHARS = 4
MAX_LONG_SENTENCE_EXAMPLES = 3

TextPart = Tuple[str, str]  # (label for messages, LLM-written text)


def check_post(
    post: PostDraft,
    keep_entries: List[DigestEntry],
    all_entries: List[DigestEntry],
    details: Dict[str, PaperDetails],
) -> VerifyResult:
    """Run every deterministic check; return hard errors and warnings (deduplicated)."""
    errors: List[str] = []
    warnings: List[str] = []
    rendered = render_markdown(post, keep_entries)
    parts = _llm_parts(post, keep_entries)

    _check_links(rendered, parts, keep_entries, all_entries, errors)
    flagged_names = _check_author_names(parts, keep_entries, details, errors)
    _check_structure(rendered, post, keep_entries, errors)
    _check_contribution_sentences(post, keep_entries, errors)
    _check_banned_phrases(rendered, parts, errors)
    _check_title_number(post.title, len(keep_entries), errors)

    _warn_surnames(parts, keep_entries, details, flagged_names, warnings)
    _warn_style(post, keep_entries, warnings)

    return VerifyResult(errors=_dedupe(errors), warnings=_dedupe(warnings))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dedupe(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _short(title: str, limit: int = 60) -> str:
    title = " ".join(title.split())
    return title if len(title) <= limit else title[: limit - 1].rstrip() + "…"


def _body_label(entry: DigestEntry) -> str:
    return f'body of "{_short(entry.title)}"'


def _llm_parts(post: PostDraft, keep_entries: List[DigestEntry]) -> List[TextPart]:
    parts: List[TextPart] = [("title", post.title or ""), ("intro", post.intro or "")]
    keep_urls = {e.url for e in keep_entries}
    for entry in keep_entries:
        parts.append((_body_label(entry), post.bodies.get(entry.url, "") or ""))
    for url, body in post.bodies.items():
        if url not in keep_urls:
            parts.append((f'body for "{url}"', body or ""))
    return parts


def _fold(text: str) -> str:
    """Strip accents so "Nicolò" also matches "Nicolo"."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _normalize_quotes(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def _paragraphs(rendered: str) -> List[str]:
    """Content paragraphs: no "# " title line and no image-only paragraphs."""
    paras = [p.strip() for p in PARAGRAPH_SPLIT_RE.split(rendered.strip()) if p.strip()]
    return [p for p in paras if not p.startswith("# ") and not IMAGE_LINE_RE.match(p)]


def _count_paragraphs(text: str) -> int:
    return len([p for p in PARAGRAPH_SPLIT_RE.split(text.strip()) if p.strip()])


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(" ".join(text.split())) if s.strip()]


# ---------------------------------------------------------------------------
# Hard errors
# ---------------------------------------------------------------------------


def _check_links(
    rendered: str,
    parts: List[TextPart],
    keep_entries: List[DigestEntry],
    all_entries: List[DigestEntry],
    errors: List[str],
) -> None:
    report_urls = {e.url for e in all_entries}
    targets = LINK_TARGET_RE.findall(rendered)

    for target in targets:
        if target not in report_urls:
            errors.append(f'link "{target}" does not exactly match any paper URL in the digest report')

    for entry in keep_entries:
        count = sum(1 for t in targets if t == entry.url)
        if count != 1:
            errors.append(
                f'"{_short(entry.title)}" is linked {count} times; each KEEP paper must be linked exactly once'
            )

    for label, text in parts:
        if LINK_TARGET_RE.search(text):
            errors.append(f"{label} contains a markdown link; links are added automatically, so remove it")
        else:
            bare = BARE_URL_RE.search(text)
            if bare:
                errors.append(f'{label} contains a URL ("{bare.group(0)}"); remove it')

    keep_ids = {m.group(0) for e in keep_entries for m in ARXIV_ABS_RE.finditer(e.url)}
    for hit in ARXIV_ABS_RE.findall(rendered):
        if hit not in keep_ids:
            errors.append(f'arXiv link "{hit}" does not match the id and version of any KEEP paper')


def _name_tokens(name: str) -> List[str]:
    return [t.strip(",") for t in name.split() if t.strip(",")]


def _name_variants(name: str) -> List[List[str]]:
    """Full name, plus first + last for names with middle names or initials."""
    tokens = _name_tokens(name)
    if len(tokens) < 2:
        return []
    variants = [tokens]
    core = [t for t in tokens if t.lower() not in _NAME_SUFFIXES]
    if len(core) >= 3:
        variants.append([core[0], core[-1]])
    return variants


def _contains_tokens(text: str, tokens: List[str], ignore_case: bool) -> bool:
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(_fold(t)) for t in tokens) + r"(?!\w)"
    return re.search(pattern, _fold(text), re.IGNORECASE if ignore_case else 0) is not None


def _check_author_names(
    parts: List[TextPart],
    keep_entries: List[DigestEntry],
    details: Dict[str, PaperDetails],
    errors: List[str],
) -> Set[str]:
    """Flag full author names; return the names that were flagged."""
    flagged: Set[str] = set()
    for entry in keep_entries:
        paper = details_for(entry, details)
        if paper is None:
            continue
        for name in paper.authors:
            for variant in _name_variants(name):
                for label, text in parts:
                    if _contains_tokens(text, variant, ignore_case=True):
                        shown = " ".join(variant)
                        errors.append(
                            f'author name "{shown}" (author of "{_short(entry.title)}") appears in {label}; '
                            "never mention author names"
                        )
                        flagged.add(name)

    for label, text in parts:
        if ET_AL_RE.search(text):
            errors.append(f'{label} contains "et al."; never mention author names')
        if MAIL_MERGE_RE.search(text):
            errors.append(f'{label} uses the banned "and other authors" pattern')
    return flagged


def _check_structure(
    rendered: str, post: PostDraft, keep_entries: List[DigestEntry], errors: List[str]
) -> None:
    keep_urls = {e.url for e in keep_entries}

    if not (post.title or "").strip():
        errors.append("title is empty")
    elif "\n" in post.title.strip():
        errors.append("title must be a single line")
    if not (post.intro or "").strip():
        errors.append("intro is empty")

    for entry in keep_entries:
        if not (post.bodies.get(entry.url) or "").strip():
            errors.append(f'no body for KEEP paper "{_short(entry.title)}" ({entry.url})')
    for url in post.bodies:
        if url not in keep_urls:
            errors.append(f'body given for "{url}", which is not one of the KEEP paper URLs')

    expected = 1 + len(keep_entries)
    found = len(_paragraphs(rendered))
    if found != expected:
        errors.append(
            f"post has {found} paragraphs; expected {expected} (1 intro + {len(keep_entries)} papers)"
        )
        if (post.intro or "").strip() and _count_paragraphs(post.intro) > 1:
            errors.append("intro must be a single paragraph (no blank lines)")
        for entry in keep_entries:
            body = post.bodies.get(entry.url) or ""
            if body.strip() and _count_paragraphs(body) > 1:
                errors.append(f"{_body_label(entry)} must be a single paragraph (no blank lines)")


def _check_contribution_sentences(
    post: PostDraft, keep_entries: List[DigestEntry], errors: List[str]
) -> None:
    for entry in keep_entries:
        body = post.bodies.get(entry.url) or ""
        if body.strip() and not CONTRIBUTION_RE.search(body):
            errors.append(
                f"{_body_label(entry)} has no explicit contribution sentence "
                '(e.g. "The verification contribution is …")'
            )


def _check_banned_phrases(rendered: str, parts: List[TextPart], errors: List[str]) -> None:
    for label, text in parts:
        lowered = _normalize_quotes(text).lower()
        for phrase in BANNED_PHRASES:
            if phrase in lowered:
                errors.append(f'{label} uses the banned phrase "{phrase}"')

    paragraphs = _paragraphs(rendered)
    if paragraphs:
        last = TRAILING_CLOSERS_RE.sub("", paragraphs[-1])
        if last.endswith("?"):
            errors.append("the post ends with a question; end the last paragraph on a statement")


def _title_numbers(title: str) -> List[int]:
    tokens = re.findall(r"\d+|[a-z]+", title.lower())
    numbers: List[int] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.isdigit():
            numbers.append(int(token))
        elif token in _TENS and i + 1 < len(tokens) and tokens[i + 1] in _UNITS:
            numbers.append(_TENS[token] + _UNITS[tokens[i + 1]])
            i += 1
        elif token in NUMBER_WORDS:
            numbers.append(NUMBER_WORDS[token])
        i += 1
    return numbers


def _check_title_number(title: str, keep_count: int, errors: List[str]) -> None:
    if not (title or "").strip():
        return
    numbers = _title_numbers(title)
    if not numbers:
        errors.append(f"title must state the number of papers ({keep_count})")
    elif keep_count not in numbers:
        shown = ", ".join(str(n) for n in numbers)
        errors.append(f"title states {shown} but the post covers {keep_count} papers")


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


def _surname(name: str) -> Optional[str]:
    core = [t for t in _name_tokens(name) if t.lower() not in _NAME_SUFFIXES]
    if not core:
        return None
    surname = core[-1].strip(".")
    return surname if len(surname) >= MIN_SURNAME_CHARS and surname[:1].isupper() else None


def _warn_surnames(
    parts: List[TextPart],
    keep_entries: List[DigestEntry],
    details: Dict[str, PaperDetails],
    flagged_names: Set[str],
    warnings: List[str],
) -> None:
    for entry in keep_entries:
        paper = details_for(entry, details)
        if paper is None:
            continue
        for name in paper.authors:
            if name in flagged_names:
                continue
            surname = _surname(name)
            if not surname or _contains_tokens(entry.title, [surname], ignore_case=True):
                continue
            for label, text in parts:
                if _contains_tokens(text, [surname], ignore_case=False):
                    warnings.append(
                        f'possible author surname "{surname}" (author of "{_short(entry.title)}") '
                        f"appears in {label}"
                    )


def _warn_style(post: PostDraft, keep_entries: List[DigestEntry], warnings: List[str]) -> None:
    long_sentences: List[Tuple[str, int, str]] = []

    intro = (post.intro or "").strip()
    if re.match(r"how (do you know|can you trust)\b", intro, re.IGNORECASE):
        warnings.append("intro opens by restating the newsletter premise; preview the papers instead")
    for sentence in _sentences(intro):
        n = len(sentence.split())
        if n > MAX_SENTENCE_WORDS:
            long_sentences.append(("intro", n, sentence))

    for entry in keep_entries:
        body = (post.bodies.get(entry.url) or "").strip()
        if not body:
            continue
        label = _body_label(entry)
        words = len(body.split())
        if words < MIN_BODY_WORDS or words > MAX_BODY_WORDS:
            warnings.append(f"{label} has {words} words (target {MIN_BODY_WORDS}–{MAX_BODY_WORDS})")
        opener = re.match(r"(this paper presents|in this paper)\b", body, re.IGNORECASE)
        if opener:
            warnings.append(f'{label} starts with "{opener.group(0)}"; lead with the problem or method')
        for sentence in _sentences(body):
            n = len(sentence.split())
            if n > MAX_SENTENCE_WORDS:
                long_sentences.append((label, n, sentence))

    if long_sentences:
        examples = "; ".join(
            f'{label}: "{" ".join(s.split()[:12])} …" ({n} words)'
            for label, n, s in long_sentences[:MAX_LONG_SENTENCE_EXAMPLES]
        )
        warnings.append(
            f"{len(long_sentences)} sentence(s) exceed {MAX_SENTENCE_WORDS} words, e.g. {examples}"
        )
