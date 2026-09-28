"""Parse a weekly digest report (reports/YYYY-MM-DD.md) into DigestEntry records.

The report is rendered by templates/report.md.j2. We read only what the
Substack pipeline needs: the ranked "### N. [Title](url)" blocks with their
"Venue / source", "Summary" and "Tags" bullets, then the "## Honorable
mentions" bullets. A "## Quiet week" report has no papers and yields [].

Author lines are ignored on purpose: the post must not name authors, and the
full author list (for the verify step) comes from arXiv, not the digest.

Paper URLs are kept verbatim, because the report is the source of truth for
every link in the post.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from .models import DigestEntry

logger = logging.getLogger(__name__)

DIGEST_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")

# "### 3. [Title](url)". The title is greedy so a title that itself contains
# brackets still parses; the URL has no whitespace and closes the line.
_TOP_HEADING_RE = re.compile(r"^###\s+(?P<num>\d+)\.\s+\[(?P<title>.*)\]\((?P<url>\S+)\)\s*$")
# "- [Title](url) — venue" (the venue may be empty).
_HONORABLE_RE = re.compile(r"^-\s+\[(?P<title>.*)\]\((?P<url>\S+)\)(?:\s+—\s*(?P<venue>.*?))?\s*$")
# "- **Summary:** text"
_FIELD_RE = re.compile(r"^-\s+\*\*(?P<name>[^*]+?):\*\*\s*(?P<value>.*)$")
# Trailing "(arxiv, hf_papers)" sources group on the "Venue / source" line.
_SOURCES_SUFFIX_RE = re.compile(r"\s*\([^()]*\)\s*$")

_ARXIV_ID = r"(?P<id>\d{4}\.\d{4,5})(?P<ver>v\d+)?"
_ARXIV_URL_RE = re.compile(
    r"^https?://(?:www\.|export\.)?arxiv\.org/(?:abs|pdf|html)/" + _ARXIV_ID + r"(?:\.pdf)?/?(?:[?#].*)?$",
    re.IGNORECASE,
)
_HF_URL_RE = re.compile(
    r"^https?://(?:www\.)?huggingface\.co/papers/" + _ARXIV_ID + r"/?(?:[?#].*)?$",
    re.IGNORECASE,
)


def find_report(reports_dir: Path, date: Optional[str] = None) -> Path:
    """Return the digest report for `date`, or the newest digest when `date` is None.

    Only date-only names (YYYY-MM-DD.md) are digests; finished posts such as
    YYYY-MM-DD-substack-post-ste100.md are never returned.
    """
    reports_dir = Path(reports_dir)
    if date is not None:
        path = reports_dir / f"{date}.md"
        if not path.is_file():
            raise FileNotFoundError(f"No digest report for {date}: {path}")
        return path

    candidates = (
        sorted(p for p in reports_dir.iterdir() if p.is_file() and DIGEST_NAME_RE.match(p.name))
        if reports_dir.is_dir()
        else []
    )
    if not candidates:
        raise FileNotFoundError(f"No digest reports (YYYY-MM-DD.md) found in {reports_dir}")
    return candidates[-1]  # ISO dates sort lexicographically


def extract_arxiv_id(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (bare_id, version) for arXiv abs/pdf/html and Hugging Face paper URLs.

    arxiv.org/abs/2609.28614v1 -> ("2609.28614", "v1");
    huggingface.co/papers/2609.00581 -> ("2609.00581", None);
    anything else (including old-style ids like hep-th/9901001) -> (None, None).
    """
    url = (url or "").strip()
    for pattern in (_ARXIV_URL_RE, _HF_URL_RE):
        m = pattern.match(url)
        if m:
            return m.group("id"), m.group("ver")
    return None, None


def _new_entry(rank: int, section: str, title: str, url: str, venue: str = "") -> DigestEntry:
    arxiv_id, version = extract_arxiv_id(url)
    return DigestEntry(
        rank=rank,
        section=section,
        title=title.strip(),
        url=url,
        arxiv_id=arxiv_id,
        version=version,
        venue=venue.strip(),
    )


def _apply_field(entry: DigestEntry, name: str, value: str) -> Optional[str]:
    """Store one "- **Name:** value" bullet; return the field key if it may continue."""
    key = name.strip().lower()
    value = value.strip()
    if key == "venue / source":
        entry.venue = _SOURCES_SUFFIX_RE.sub("", value).strip()
        return None
    if key == "summary":
        entry.digest_summary = value
        return "summary"
    if key == "tags":
        entry.tags = [t.strip() for t in value.split(",") if t.strip()]
        return None
    return None  # Authors, Reputation, ... are not needed


def parse_report(path: Path) -> List[DigestEntry]:
    """Parse a digest report into ranked entries: top papers, then honorable mentions."""
    text = Path(path).read_text(encoding="utf-8")

    entries: List[DigestEntry] = []
    section: Optional[str] = None  # "top" | "honorable" | None
    current: Optional[DigestEntry] = None
    continuing: Optional[str] = None  # field whose value may wrap onto the next line

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("## "):
            heading = stripped[3:].strip().lower()
            current, continuing = None, None
            if heading.startswith("quiet week"):
                logger.info("Quiet-week report %s: no papers", path)
                return []
            if heading.startswith("top papers"):
                section = "top"
            elif heading.startswith("honorable mentions"):
                section = "honorable"
            else:
                section = None
            continue

        if section == "top":
            m = _TOP_HEADING_RE.match(stripped)
            if m:
                current = _new_entry(len(entries) + 1, "top", m.group("title"), m.group("url"))
                entries.append(current)
                continuing = None
                continue
            if current is None:
                continue
            fm = _FIELD_RE.match(stripped)
            if fm:
                continuing = _apply_field(current, fm.group("name"), fm.group("value"))
                continue
            if not stripped or stripped.startswith("#") or stripped.startswith("- "):
                continuing = None
                continue
            if continuing == "summary":  # wrapped summary text
                current.digest_summary = f"{current.digest_summary} {stripped}".strip()
            continue

        if section == "honorable":
            m = _HONORABLE_RE.match(stripped)
            if m:
                entries.append(
                    _new_entry(len(entries) + 1, "honorable", m.group("title"), m.group("url"), m.group("venue") or "")
                )
            continue

    top = sum(1 for e in entries if e.section == "top")
    logger.info("Parsed %s: %d top papers, %d honorable mentions", path, top, len(entries) - top)
    return entries
