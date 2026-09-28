"""Prompt construction for the Substack post writer.

WRITING_SUBSTACK_POST.md is loaded at runtime, so that file stays the single
source of truth for the writing rules. This module only adds a short
pipeline-mode preamble (abstracts are already fetched, no tools, answer in
JSON) and formats the per-paper material the LLM calls need.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import PROJECT_ROOT
from .models import DigestEntry, PaperDetails, Verdict

logger = logging.getLogger(__name__)

RULES_FILENAME = "WRITING_SUBSTACK_POST.md"
STYLE_EXAMPLE_SUFFIX = "-substack-post-ste100.md"
_STYLE_EXAMPLE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})" + re.escape(STYLE_EXAMPLE_SUFFIX) + r"$")

MAX_ABSTRACT_CHARS = 3000
MAX_CAPTION_CHARS = 400
ABSTRACT_UNAVAILABLE = "abstract unavailable — judge from title/summary, lean DROP"

JSON_RETRY_PREFIX = (
    "Your previous response was not valid JSON matching the required schema. "
    "Respond again with ONLY the JSON object.\n\n"
)

_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_H3_RE = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Loading the rules doc and the style example
# ---------------------------------------------------------------------------


def load_rules(path: Optional[Path] = None) -> str:
    """Return the full text of WRITING_SUBSTACK_POST.md."""
    rules_path = path or (PROJECT_ROOT / RULES_FILENAME)
    return rules_path.read_text(encoding="utf-8")


def load_style_example(reports_dir: Path, exclude_date: Optional[str] = None) -> Optional[str]:
    """Return the newest reports/<date>-substack-post-ste100.md, skipping exclude_date.

    Returns None when no such post exists.
    """
    candidates: List[Tuple[str, Path]] = []
    for path in Path(reports_dir).glob("*" + STYLE_EXAMPLE_SUFFIX):
        match = _STYLE_EXAMPLE_RE.match(path.name)
        if not match:
            continue
        post_date = match.group(1)
        if exclude_date and post_date == exclude_date:
            continue
        candidates.append((post_date, path))
    if not candidates:
        return None
    _, newest = max(candidates)
    return newest.read_text(encoding="utf-8")


def extract_section(markdown: str, number: int, keyword: str = "") -> Optional[str]:
    """Return one level-2 section ("## <number>. ...") of the rules doc, heading included.

    The section is found by its markdown heading: first by its number, then by
    ``keyword`` appearing in the heading text. It runs up to the next level-2
    heading. Returns None when no heading matches.
    """
    headings = list(_H2_RE.finditer(markdown))
    chosen = None
    for i, match in enumerate(headings):
        if re.match(rf"{number}\.\s", match.group(1)):
            chosen = i
            break
    if chosen is None and keyword:
        for i, match in enumerate(headings):
            if keyword.lower() in match.group(1).lower():
                chosen = i
                break
    if chosen is None:
        return None
    start = headings[chosen].start()
    end = headings[chosen + 1].start() if chosen + 1 < len(headings) else len(markdown)
    return markdown[start:end].strip()


def _drop_subsections(section: str, keyword: str) -> str:
    """Remove every level-3 subsection whose heading contains ``keyword``."""
    headings = list(_H3_RE.finditer(section))
    if not headings:
        return section
    pieces: List[str] = []
    cursor = 0
    for i, match in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(section)
        if keyword.lower() in match.group(1).lower():
            pieces.append(section[cursor : match.start()])
            cursor = end
    pieces.append(section[cursor:])
    return "".join(pieces).strip()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def details_for(entry: DigestEntry, details: Dict[str, PaperDetails]) -> Optional[PaperDetails]:
    """Look up a paper's arXiv details by report URL, falling back to the bare arXiv id."""
    found = details.get(entry.url)
    if found is None and entry.arxiv_id:
        found = details.get(entry.arxiv_id)
    return found


def _abstract_text(entry: DigestEntry, details: Dict[str, PaperDetails]) -> str:
    paper = details_for(entry, details)
    abstract = " ".join((paper.abstract if paper else "").split())
    if not abstract:
        return ABSTRACT_UNAVAILABLE
    if len(abstract) > MAX_ABSTRACT_CHARS:
        abstract = abstract[:MAX_ABSTRACT_CHARS].rstrip() + " …"
    return abstract


def with_json_retry_prefix(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return a copy of ``messages`` whose last user message starts with the strict JSON reminder."""
    updated = [dict(m) for m in messages]
    for message in reversed(updated):
        if message["role"] == "user":
            message["content"] = JSON_RETRY_PREFIX + message["content"]
            break
    return updated


# ---------------------------------------------------------------------------
# Filter prompt (WRITING_SUBSTACK_POST.md sections 1-2)
# ---------------------------------------------------------------------------

FILTER_PIPELINE_NOTE = (
    "You are the strict KEEP/DROP filter for a weekly newsletter post. The editor's own "
    "rules for this step follow, copied from the style guide.\n\n"
)

FILTER_INSTRUCTIONS = (
    "How to apply these rules here:\n"
    "- The real arXiv abstract is already provided for each paper. Judge the paper's CORE "
    "contribution from the abstract, not from the digest's one-line summary or tags.\n"
    "- Be strict and honest. Pipeline tags such as `evals` or `verification` do not by "
    "themselves justify KEEP. A paper whose real core contribution is a capability benchmark, "
    "an agent architecture, or an application that only touches verification, validation, or "
    "uncertainty quantification tangentially is DROP.\n"
    "- Expect roughly 25–35% of the full paper list to be KEEP. Papers are sent to you in "
    "batches, so one batch can have more or fewer.\n"
    "- When the abstract is unavailable, judge from the title and summary and lean DROP.\n"
    "- For KEEP papers, write `summary` as a 4–6 sentence plain-English summary of the actual "
    "method and result (the specific mechanism, names, and numbers). This is the raw material "
    "for the post. For DROP papers, `summary` is an empty string.\n"
    "- `contribution_type` is one of \"verification\", \"validation\", "
    "\"uncertainty-quantification\" for KEEP, and \"\" for DROP.\n"
    "- Respond only with the specified JSON, no other text."
)


def build_filter_system_prompt(rules: str) -> str:
    """System prompt for the KEEP/DROP filter: sections 1-2 of the rules doc plus pipeline notes."""
    parts: List[str] = []
    source = extract_section(rules, 1, "source")
    definition = extract_section(rules, 2, "filter")
    if source:
        parts.append(source)
    if definition:
        parts.append(_drop_subsections(definition, "in practice"))
    if not definition:
        logger.warning("Rules doc has no section 2 filter heading; sending the whole doc to the filter")
        parts = [rules.strip()]
    return FILTER_PIPELINE_NOTE + "\n\n".join(parts) + "\n\n" + FILTER_INSTRUCTIONS


def build_filter_user_message(batch: List[DigestEntry], details: Dict[str, PaperDetails]) -> str:
    items: List[str] = []
    for i, entry in enumerate(batch):
        section = "top paper" if entry.section == "top" else "honorable mention"
        tags = ", ".join(entry.tags) if entry.tags else "(none)"
        one_liner = entry.digest_summary.strip() or "(none)"
        items.append(
            f"{i}. Title: {entry.title}\n"
            f"   URL: {entry.url}\n"
            f"   Report position: #{entry.rank} ({section})\n"
            f"   Digest one-liner (automated, often wrong): {one_liner}\n"
            f"   Pipeline tags (automated, often wrong): {tags}\n"
            f"   Abstract: {_abstract_text(entry, details)}"
        )
    papers_block = "\n\n".join(items)
    return (
        f"Papers:\n\n{papers_block}\n\n"
        'Respond with ONLY this JSON shape: {"results": [{"index": <int>, '
        '"verdict": "KEEP" | "DROP", "rationale": "<one line>", '
        '"summary": "<4-6 sentences for KEEP, empty for DROP>", '
        '"contribution_type": "verification" | "validation" | "uncertainty-quantification" | ""}, ...]} '
        "— one entry per paper, indices matching the list above."
    )


# ---------------------------------------------------------------------------
# Figure choice prompt
# ---------------------------------------------------------------------------

FIGURE_SYSTEM_PROMPT = (
    "You pick the main figure of research papers for a newsletter. For each paper, choose the "
    "figure that best shows how the method works: the overview, architecture, pipeline, "
    "framework, workflow, or method diagram. Prefer it over results plots, tables, example "
    "outputs, and ablations. When no caption describes such a diagram, choose the figure that "
    "best explains the paper's main idea. Respond only with the specified JSON, no other text."
)


def build_figure_user_message(papers: List[List[str]]) -> str:
    blocks: List[str] = []
    for p, captions in enumerate(papers):
        lines = [f"Paper {p}:"]
        for f, caption in enumerate(captions):
            text = " ".join(caption.split()) or "(no caption)"
            if len(text) > MAX_CAPTION_CHARS:
                text = text[:MAX_CAPTION_CHARS].rstrip() + " …"
            lines.append(f"  [{f}] {text}")
        blocks.append("\n".join(lines))
    return (
        "Figure captions per paper (figure indices are 0-based):\n\n"
        + "\n\n".join(blocks)
        + '\n\nRespond with ONLY this JSON shape: {"choices": [{"paper": <int>, "figure": <int>}, ...]} '
        "— one entry per paper, using the indices shown above."
    )


# ---------------------------------------------------------------------------
# Writer prompt (WRITING_SUBSTACK_POST.md sections 3-5)
# ---------------------------------------------------------------------------

WRITER_PREAMBLE = """You write this week's Substack post in an automated pipeline. The editor's full writing rules follow after this preamble. Obey sections 3, 4, and 5 exactly.

Pipeline mode — these points override the rules document where they differ:
- The filtering (sections 1 and 2) is already done. The user message gives only the KEEP papers, each with its real arXiv abstract and a summary. The abstracts and summaries are already provided below — do not re-filter, and write about every paper given.
- Ignore every instruction about Agents, WebFetch, skills, saving files, and the mechanics of sections 6 and 7. You cannot run tools. Automated checks run on your answer, and failures come back to you for a rewrite.
- Never mention author names — not one, not even a surname. Never write "et al." and never write "and other authors". Use "this paper", the benchmark, or the method's own name as the subject.
- Return ONLY a JSON object with this exact shape:
  {"title": "<post title>", "intro": "<one intro paragraph>", "papers": [{"url": "<paper URL copied exactly>", "body": "<one paragraph>"}]}
  with exactly one entry in "papers" per paper, in the order given.
- "body" must NOT start with the paper title or a link. The pipeline adds "**[Title](url)** — " in front of each body. Start the body with the concrete problem or method, not "This paper presents" or "In this paper".
- Never put a markdown link or a URL in the title, the intro, or any body.
- Each body must contain one explicit sentence naming the paper's specific contribution, in the form "The verification contribution is …", "The validation-methodology contribution is …", or "The uncertainty-quantification contribution is …".
- Each body is one paragraph (no blank lines) of about 80–200 words. The intro is one paragraph.
- The title must contain the number of papers, as a digit or a word, and be concrete.
- Do not end the post with a question and do not add a closing section."""

STYLE_EXAMPLE_HEADER = (
    "# A previously approved post\n\n"
    "The post below is a previously approved post — match its register and structure, do not "
    "copy its content. Its papers are not this week's papers."
)


def build_writer_system_prompt(rules: str, style_example: Optional[str] = None) -> str:
    """Pipeline preamble + the FULL rules doc + (optionally) an approved post as a style example."""
    parts = [WRITER_PREAMBLE, "# The editor's writing rules (WRITING_SUBSTACK_POST.md)\n\n" + rules.strip()]
    if style_example and style_example.strip():
        parts.append(
            STYLE_EXAMPLE_HEADER
            + "\n\n<approved_post>\n"
            + style_example.strip()
            + "\n</approved_post>"
        )
    return "\n\n---\n\n".join(parts)


def build_writer_user_message(
    keep_entries: List[DigestEntry],
    verdicts: Dict[str, Verdict],
    details: Dict[str, PaperDetails],
) -> str:
    n = len(keep_entries)
    items: List[str] = []
    for i, entry in enumerate(keep_entries, start=1):
        verdict = verdicts.get(entry.url)
        contribution = (verdict.contribution_type if verdict else "") or "(unspecified)"
        summary = (verdict.summary if verdict else "").strip() or "(none)"
        items.append(
            f"{i}. Title: {entry.title}\n"
            f"   URL: {entry.url}\n"
            f"   Contribution type: {contribution}\n"
            f"   Summary: {summary}\n"
            f"   Abstract: {_abstract_text(entry, details)}"
        )
    return (
        f"This week's KEEP papers, in report order ({n} papers):\n\n"
        + "\n\n".join(items)
        + f"\n\nWrite the post now. The title must state the number {n}. Return exactly {n} "
        'entries in "papers", in the order above, each with its "url" copied exactly. '
        "Return ONLY the JSON object."
    )


def build_rewrite_message(errors: List[str]) -> str:
    """Follow-up user message after a draft fails the deterministic checks."""
    error_lines = "\n".join(f"- {e}" for e in errors)
    return (
        "Your post failed these automated checks:\n"
        f"{error_lines}\n\n"
        "Rewrite the post to fix every failure listed above. Keep the parts that passed. "
        "Return the complete JSON object again, with the same shape "
        '({"title", "intro", "papers": [{"url", "body"}]}), and ONLY the JSON object.'
    )


def dump_post_json(title: str, intro: str, papers: List[Dict[str, str]]) -> str:
    """Canonical JSON of a post attempt, used when sending a draft back for a rewrite."""
    return json.dumps({"title": title, "intro": intro, "papers": papers}, ensure_ascii=False, indent=2)
