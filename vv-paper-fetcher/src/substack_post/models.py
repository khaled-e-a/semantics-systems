"""Records passed between the digest → Substack draft pipeline stages.

Everything is keyed by the paper URL exactly as it appears in the digest
report, because that URL is the source of truth for links in the post.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DigestEntry:
    """One paper parsed from reports/YYYY-MM-DD.md."""

    rank: int  # 1-based position across the whole report (top papers, then honorable mentions)
    section: str  # "top" | "honorable"
    title: str
    url: str  # verbatim from the report
    arxiv_id: Optional[str] = None  # bare id without version, e.g. "2609.28614"
    version: Optional[str] = None  # e.g. "v1" when the report URL carries one
    venue: str = ""
    digest_summary: str = ""  # empty for honorable mentions
    tags: List[str] = field(default_factory=list)


@dataclass
class PaperDetails:
    """Real metadata fetched from arXiv for one DigestEntry."""

    arxiv_id: str
    version: Optional[str]  # version actually returned by arXiv, e.g. "v2"
    title: str
    abstract: str
    authors: List[str]  # full author list
    license_url: Optional[str] = None  # filled for KEEP papers only


@dataclass
class Verdict:
    """LLM KEEP/DROP decision against the strict VV/UQ definition."""

    url: str
    keep: bool
    rationale: str
    summary: str = ""  # 4-6 sentence plain-English summary, KEEP only
    contribution_type: str = ""  # "verification" | "validation" | "uncertainty-quantification" | "" for DROP


@dataclass
class Figure:
    """A paper's main diagram, downloaded and normalized to a local image file."""

    url: str  # paper URL (key)
    path: Path  # local PNG/JPEG file
    figure_number: str  # e.g. "1"
    credit_caption: str  # e.g. 'Figure 1 from “Title” (arXiv:2609.28614)'
    source: str  # "html" | "pdf"
    openly_licensed: bool  # CC BY / CC BY-SA / CC0
    width: int
    height: int
    mime: str  # "image/png" | "image/jpeg"


@dataclass
class PostDraft:
    """LLM-written post before images/links are assembled."""

    title: str
    intro: str
    bodies: Dict[str, str]  # paper URL -> paragraph body WITHOUT the "**[Title](url)** — " prefix


@dataclass
class VerifyResult:
    """Outcome of the deterministic WRITING_SUBSTACK_POST.md section 7 checks."""

    errors: List[str] = field(default_factory=list)  # hard failures → rewrite
    warnings: List[str] = field(default_factory=list)  # reported in the email only

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class UploadedImage:
    """An image hosted on Substack's CDN after upload."""

    url: str
    width: int
    height: int
    bytes: int
    content_type: str


@dataclass
class DraftResult:
    """A created or updated Substack draft."""

    draft_id: int
    edit_url: str  # https://<pub>.substack.com/publish/post/<draft_id>
    created: bool  # False when an existing draft was updated
