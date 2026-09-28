"""verify.check_post against the hand-approved 2026-09-07 post, then with faults injected."""
import re
from copy import deepcopy
from typing import Dict, List, Tuple

import pytest

from src.config import PROJECT_ROOT
from src.substack_post.models import DigestEntry, PaperDetails, PostDraft
from src.substack_post.verify import check_post

APPROVED_POST = PROJECT_ROOT / "reports" / "2026-09-07-substack-post-ste100.md"
SOURCE_REPORT = PROJECT_ROOT / "reports" / "2026-09-07.md"

PARAGRAPH_RE = re.compile(r"^\*\*\[(?P<title>.+?)\]\((?P<url>\S+?)\)\*\* — (?P<body>.*)$", re.DOTALL)


def _load_approved() -> Tuple[PostDraft, List[DigestEntry]]:
    paragraphs = re.split(r"\n\s*\n", APPROVED_POST.read_text(encoding="utf-8").strip())
    assert paragraphs[0].startswith("# ")
    entries: List[DigestEntry] = []
    bodies: Dict[str, str] = {}
    for rank, paragraph in enumerate(paragraphs[2:], start=1):
        match = PARAGRAPH_RE.match(paragraph)
        assert match, paragraph[:80]
        entries.append(DigestEntry(rank=rank, section="top", title=match["title"], url=match["url"]))
        bodies[match["url"]] = match["body"]
    post = PostDraft(title=paragraphs[0][2:], intro=paragraphs[1], bodies=bodies)
    return post, entries


@pytest.fixture
def approved():
    return _load_approved()


def _errors(post, entries, all_entries=None, details=None) -> List[str]:
    return check_post(post, entries, all_entries or entries, details or {}).errors


def _edit_body(post: PostDraft, entry: DigestEntry, new_body: str) -> PostDraft:
    edited = deepcopy(post)
    edited.bodies[entry.url] = new_body
    return edited


def test_approved_post_passes(approved):
    post, entries = approved
    assert len(entries) == 15
    result = check_post(post, entries, entries, {})
    assert result.ok, result.errors


def test_approved_post_passes_with_real_author_lists(approved):
    """The digest's author lines (first three authors) must not trigger author-name errors."""
    post, entries = approved
    report = SOURCE_REPORT.read_text(encoding="utf-8")
    details: Dict[str, PaperDetails] = {}
    for match in re.finditer(r"### \d+\. \[.+?\]\((\S+?)\)\n\n- \*\*Authors:\*\* (.+)", report):
        url, authors = match.group(1), match.group(2)
        names = [a.strip() for a in authors.split(",") if a.strip() and a.strip() != "et al."]
        details[url] = PaperDetails(arxiv_id="x", version="v1", title="", abstract="", authors=names)
    assert sum(1 for e in entries if e.url in details) >= 10
    result = check_post(post, entries, entries, details)
    assert result.ok, result.errors


# --- links ---------------------------------------------------------------------------


def test_keep_link_with_wrong_version_is_an_error(approved):
    post, entries = approved
    report_entries = deepcopy(entries)
    bad = deepcopy(entries)
    wrong_url = bad[0].url.replace("v1", "v2")
    post = deepcopy(post)
    post.bodies[wrong_url] = post.bodies.pop(bad[0].url)
    bad[0].url = wrong_url
    errors = _errors(post, bad, all_entries=report_entries)
    assert any(wrong_url in e and "does not exactly match" in e for e in errors), errors


def test_arxiv_link_in_body_with_wrong_version_is_an_error(approved):
    post, entries = approved
    entry = entries[0]
    post = _edit_body(post, entry, post.bodies[entry.url] + " The code is at arxiv.org/abs/2609.03460v2 too.")
    errors = _errors(post, entries)
    assert any("2609.03460v2" in e and "id and version" in e for e in errors), errors
    assert any("contains a URL" in e for e in errors), errors


def test_extra_link_in_body_is_an_error(approved):
    post, entries = approved
    entry = entries[3]
    post = _edit_body(post, entry, post.bodies[entry.url] + " See [the code](https://github.com/x/agentscope).")
    errors = _errors(post, entries)
    assert any("contains a markdown link" in e for e in errors), errors
    assert any("https://github.com/x/agentscope" in e for e in errors), errors


def test_link_to_report_paper_inside_body_is_an_error(approved):
    """Even an exact report URL is an error inside a body: it would double-link a paper."""
    post, entries = approved
    other = entries[1]
    post = _edit_body(post, entries[0], post.bodies[entries[0].url] + f" Compare [this]({other.url}).")
    errors = _errors(post, entries)
    assert any("linked 2 times" in e for e in errors), errors
    assert any("contains a markdown link" in e for e in errors), errors


# --- author names --------------------------------------------------------------------


def test_author_full_name_is_an_error(approved):
    post, entries = approved
    entry = entries[0]
    details = {
        entry.url: PaperDetails(
            arxiv_id="2609.03460", version="v1", title=entry.title, abstract="",
            authors=["Qing Zhang", "Yifei Huang"],
        )
    }
    post = _edit_body(post, entry, "Qing Zhang and a team build an interface. " + post.bodies[entry.url])
    errors = _errors(post, entries, details=details)
    assert any('"Qing Zhang"' in e for e in errors), errors


def test_author_name_matches_case_insensitively_and_without_middle_initial_or_accents(approved):
    post, entries = approved
    entry = entries[1]
    details = {
        entry.url: PaperDetails(
            arxiv_id="2609.04173", version="v1", title=entry.title, abstract="",
            authors=["Vilém M. Zouhar"],
        )
    }
    post = _edit_body(post, entry, post.bodies[entry.url] + " VILEM ZOUHAR led the effort.")
    errors = _errors(post, entries, details=details)
    assert any("Zouhar" in e and "never mention author names" in e for e in errors), errors


def test_et_al_is_an_error(approved):
    post, entries = approved
    entry = entries[2]
    post = _edit_body(post, entry, "Lu et al. adapt grounded theory. " + post.bodies[entry.url])
    errors = _errors(post, entries)
    assert any("et al." in e for e in errors), errors


def test_mail_merge_and_other_authors_is_an_error(approved):
    post, entries = approved
    post = deepcopy(post)
    post.intro = "Several researchers and other authors build tools. " + post.intro
    errors = _errors(post, entries)
    assert any("and other authors" in e for e in errors), errors


def test_surname_only_is_a_warning_not_an_error(approved):
    post, entries = approved
    enoki_entry = next(e for e in entries if e.title.startswith("Enoki"))
    first = entries[0]
    details = {
        first.url: PaperDetails(arxiv_id="a", version="v1", title="", abstract="", authors=["Mei Stackelberg"]),
        enoki_entry.url: PaperDetails(arxiv_id="b", version="v1", title="", abstract="", authors=["Taro Enoki"]),
    }
    post = _edit_body(post, first, post.bodies[first.url] + " Stackelberg calls this a veto.")
    result = check_post(post, entries, entries, details)
    assert result.ok, result.errors
    assert any('"Stackelberg"' in w for w in result.warnings), result.warnings
    # "Enoki" is a method name that also appears in its own paper's title: no warning.
    assert not any('"Enoki"' in w for w in result.warnings), result.warnings


# --- structure -----------------------------------------------------------------------


def test_missing_body_is_an_error(approved):
    post, entries = approved
    post = deepcopy(post)
    dropped = entries[5]
    del post.bodies[dropped.url]
    errors = _errors(post, entries)
    assert any("no body for KEEP paper" in e and dropped.url in e for e in errors), errors


def test_body_split_into_two_paragraphs_breaks_paragraph_count(approved):
    post, entries = approved
    entry = entries[4]
    body = post.bodies[entry.url]
    post = _edit_body(post, entry, body.replace(" This paper tests", "\n\nThis paper tests", 1))
    errors = _errors(post, entries)
    assert any("post has 17 paragraphs; expected 16" in e for e in errors), errors
    assert any("must be a single paragraph" in e for e in errors), errors


def test_body_for_non_keep_paper_is_an_error(approved):
    post, entries = approved
    keep = entries[:-1]
    post = deepcopy(post)
    post.title = post.title.replace("15", "14")
    errors = _errors(post, keep, all_entries=entries)
    assert any("not one of the KEEP paper URLs" in e for e in errors), errors


def test_missing_contribution_sentence_is_an_error(approved):
    post, entries = approved
    entry = entries[1]
    body = post.bodies[entry.url]
    assert "The verification contribution is this:" in body
    post = _edit_body(post, entry, body.replace("The verification contribution is this:", "In short,"))
    errors = _errors(post, entries)
    assert any("Last Translation Benchmark" in e and "contribution sentence" in e for e in errors), errors


# --- banned phrases ------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    ["This is a game-changer.", "Here’s why this matters.", "In today's rapidly evolving field, tests fail.",
     "Let us delve into the method.", "It is a cutting edge tool."],
)
def test_banned_phrase_is_an_error(approved, phrase):
    post, entries = approved
    entry = entries[6]
    post = _edit_body(post, entry, phrase + " " + post.bodies[entry.url])
    errors = _errors(post, entries)
    assert any("banned phrase" in e for e in errors), errors


def test_trailing_question_is_an_error(approved):
    post, entries = approved
    last = entries[-1]
    post = _edit_body(post, last, post.bodies[last.url] + " What will your team check next?")
    errors = _errors(post, entries)
    assert any("ends with a question" in e for e in errors), errors


def test_question_mid_post_is_fine(approved):
    """The approved post itself has a question inside a paragraph (ClaimReceipt)."""
    post, entries = approved
    assert any("?" in body for body in post.bodies.values())
    assert check_post(post, entries, entries, {}).ok


# --- title number --------------------------------------------------------------------


def test_title_number_mismatch_is_an_error(approved):
    post, entries = approved
    post = deepcopy(post)
    post.title = "Fourteen Ways to Verify, Validate, and Calibrate AI Systems"
    errors = _errors(post, entries)
    assert any("title states 14" in e and "15 papers" in e for e in errors), errors


def test_title_without_number_is_an_error(approved):
    post, entries = approved
    post = deepcopy(post)
    post.title = "Ways to Verify, Validate, and Calibrate AI Systems"
    errors = _errors(post, entries)
    assert any("title must state the number of papers" in e for e in errors), errors


@pytest.mark.parametrize("title", ["Fifteen Ways to Check AI Systems", "15 Checks for AI Output"])
def test_title_number_as_word_or_digit_passes(approved, title):
    post, entries = approved
    post = deepcopy(post)
    post.title = title
    assert _errors(post, entries) == []


# --- warnings ------------------------------------------------------------------------


def test_style_warnings(approved):
    post, entries = approved
    post = deepcopy(post)
    post.intro = "How do you know an AI system tells the truth? " + post.intro
    post.bodies[entries[0].url] = "This paper presents a tool. The verification contribution is a check."
    result = check_post(post, entries, entries, {})
    assert result.ok, result.errors
    joined = "\n".join(result.warnings)
    assert "restating the newsletter premise" in joined
    assert "This paper presents" in joined
    assert "words (target 80–200)" in joined
    assert "exceed 25 words" in joined
