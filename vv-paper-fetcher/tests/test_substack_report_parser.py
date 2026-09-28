import re
from pathlib import Path

import pytest

from src.substack_post.report_parser import DIGEST_NAME_RE, extract_arxiv_id, find_report, parse_report

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
DIGESTS = sorted(p for p in REPORTS_DIR.glob("2026-*.md") if DIGEST_NAME_RE.match(p.name))


def _expected_counts(text: str):
    m = re.search(r"^## Top papers \((\d+)\)", text, re.MULTILINE)
    top = int(m.group(1)) if m else 0
    honorable = 0
    if "## Honorable mentions" in text:
        tail = text.split("## Honorable mentions", 1)[1]
        honorable = sum(1 for line in tail.splitlines() if line.startswith("- ["))
    return top, honorable


def test_committed_digests_exist():
    assert len(DIGESTS) >= 8
    assert all("substack-post" not in p.name for p in DIGESTS)


@pytest.mark.parametrize("path", DIGESTS, ids=lambda p: p.name)
def test_parse_every_committed_digest(path):
    text = path.read_text(encoding="utf-8")
    expected_top, expected_honorable = _expected_counts(text)

    entries = parse_report(path)

    top = [e for e in entries if e.section == "top"]
    honorable = [e for e in entries if e.section == "honorable"]
    assert len(top) == expected_top
    assert len(honorable) == expected_honorable
    # top papers first, then honorable mentions; ranks run 1..n across both
    assert entries == top + honorable
    assert [e.rank for e in entries] == list(range(1, len(entries) + 1))

    for e in entries:
        assert f"]({e.url})" in text  # URL kept verbatim
        assert e.title and not e.title.startswith("[")
        assert e.arxiv_id is not None
    for e in top:
        assert e.digest_summary
        assert e.tags
        assert e.venue and not e.venue.endswith(")")
    for e in honorable:
        assert e.digest_summary == "" and e.tags == []


def test_digest_2026_09_28_details():
    entries = parse_report(REPORTS_DIR / "2026-09-28.md")
    first = entries[0]
    assert first.rank == 1 and first.section == "top"
    assert first.title.startswith("BioEVAL: A global, multi-institutional benchmark")
    assert first.url == "http://arxiv.org/abs/2609.30489v1"
    assert (first.arxiv_id, first.version) == ("2609.30489", "v1")
    assert first.venue == "cs.AI"
    assert first.tags == ["evals", "scientific-ai"]
    assert first.digest_summary.startswith("Describes BioEVAL")

    jev = entries[4]
    assert jev.title.startswith("Just Ask Jev")
    assert jev.venue == "cs.AI"  # "(arxiv, hf_papers)" sources stripped
    assert jev.tags == ["evals", "uncertainty-quantification"]

    mention = entries[20]
    assert mention.section == "honorable" and mention.rank == 21
    assert mention.url == "http://arxiv.org/abs/2609.30454v1"
    assert mention.venue == "cs.LG"
    assert entries[22].version == "v2"  # .../2609.27690v2


def test_hugging_face_digest_ids():
    entries = parse_report(REPORTS_DIR / "2026-09-14.md")
    assert len(entries) == 12
    first = entries[0]
    assert first.url == "https://huggingface.co/papers/2609.11115"
    assert (first.arxiv_id, first.version) == ("2609.11115", None)
    assert first.venue == "HF Daily Papers"
    assert all(e.url.startswith("https://huggingface.co/papers/") for e in entries)


QUIET_WEEK = """# VV / UQ / Evals Paper Digest — 2026-10-01 to 2026-10-08

_Generated 2026-10-08 11:49 UTC_

Collected: arxiv=900, hf_papers=40, openreview=0, dblp=0
LLM triage batches: 25/25 succeeded

## Quiet week

No papers matched the relevance threshold this week. That's a valid outcome, not an error.
"""


def test_quiet_week_report_is_empty(tmp_path):
    path = tmp_path / "2026-10-08.md"
    path.write_text(QUIET_WEEK, encoding="utf-8")
    assert parse_report(path) == []


EDGE_CASES = """# VV / UQ / Evals Paper Digest — 2026-10-01 to 2026-10-08

## Top papers (2)

### 1. [$τ^τ$-Bench: A [Draft] Benchmark](http://arxiv.org/abs/2609.04611v1)

- **Authors:** A. Person, B. Person
- **Venue / source:** ICLR 2027 (poster) (openreview, arxiv)
- **Reputation:** max author h-index 3, paper citations 0
- **Summary:** First line of a summary
that wraps onto a second line.
- **Tags:** evals,  verification ,

### 2. [An OpenReview Paper](https://openreview.net/forum?id=AbC123)

- **Venue / source:**  (dblp)
- **Summary:** No arXiv id here.
- **Tags:** formal-methods

## Honorable mentions

- [Mention Without Venue](https://arxiv.org/pdf/2609.00001v3) —
- [Mention With Venue](http://arxiv.org/abs/2609.00002) — stat.ML
"""


def test_edge_cases(tmp_path):
    path = tmp_path / "2026-10-08.md"
    path.write_text(EDGE_CASES, encoding="utf-8")
    entries = parse_report(path)

    assert [e.rank for e in entries] == [1, 2, 3, 4]
    tau, openreview, no_venue, with_venue = entries
    assert tau.title == "$τ^τ$-Bench: A [Draft] Benchmark"
    assert tau.venue == "ICLR 2027 (poster)"
    assert tau.digest_summary == "First line of a summary that wraps onto a second line."
    assert tau.tags == ["evals", "verification"]

    assert openreview.url == "https://openreview.net/forum?id=AbC123"
    assert (openreview.arxiv_id, openreview.version) == (None, None)
    assert openreview.venue == ""

    assert no_venue.section == "honorable" and no_venue.venue == ""
    assert (no_venue.arxiv_id, no_venue.version) == ("2609.00001", "v3")
    assert with_venue.venue == "stat.ML" and with_venue.version is None


@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://arxiv.org/abs/2609.28614v1", ("2609.28614", "v1")),
        ("https://arxiv.org/abs/2609.28614", ("2609.28614", None)),
        ("https://arxiv.org/abs/2609.28614v12", ("2609.28614", "v12")),
        ("https://arxiv.org/pdf/2609.28614v2", ("2609.28614", "v2")),
        ("https://arxiv.org/pdf/2609.28614v2.pdf", ("2609.28614", "v2")),
        ("https://arxiv.org/html/2609.28614v1/", ("2609.28614", "v1")),
        ("https://www.arxiv.org/abs/1501.0001", ("1501.0001", None)),
        ("https://huggingface.co/papers/2609.00581", ("2609.00581", None)),
        ("https://huggingface.co/papers/2609.00581/", ("2609.00581", None)),
        ("https://openreview.net/forum?id=AbC123", (None, None)),
        ("http://arxiv.org/abs/hep-th/9901001v1", (None, None)),
        ("https://example.com/abs/2609.28614v1", (None, None)),
        ("", (None, None)),
    ],
)
def test_extract_arxiv_id(url, expected):
    assert extract_arxiv_id(url) == expected


def test_find_report_picks_newest_digest(tmp_path):
    for name in [
        "2026-09-21.md",
        "2026-09-28.md",
        "2026-09-30-substack-post-ste100.md",
        "2026-10-05-substack-post.md",
        "notes.md",
    ]:
        (tmp_path / name).write_text("x", encoding="utf-8")

    assert find_report(tmp_path) == tmp_path / "2026-09-28.md"
    assert find_report(tmp_path, "2026-09-21") == tmp_path / "2026-09-21.md"
    with pytest.raises(FileNotFoundError):
        find_report(tmp_path, "2026-09-30")  # only the finished post exists for that date
    with pytest.raises(FileNotFoundError):
        find_report(tmp_path / "missing")


def test_find_report_empty_dir(tmp_path):
    (tmp_path / "2026-09-07-substack-post-ste100.md").write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        find_report(tmp_path)


def test_find_report_on_committed_reports():
    newest = find_report(REPORTS_DIR)
    assert newest == DIGESTS[-1]
    assert DIGEST_NAME_RE.match(newest.name)
