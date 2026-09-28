"""LLM steps of the Substack post pipeline, via OpenRouter.

Three calls, all following the src/llm_triage.py pattern (OpenAI client on
OpenRouter, ``response_format=json_object``, one retry with a stricter
"respond again with ONLY the JSON" follow-up, then log and degrade):

- ``filter_papers``: strict KEEP/DROP against WRITING_SUBSTACK_POST.md section 2,
  in batches. A batch that fails twice becomes DROP rather than crashing the run.
- ``choose_figures``: one call that picks each paper's main diagram from its captions.
- ``write_post``: title + intro + one body per KEEP paper, checked by verify.py;
  hard errors are sent back for a rewrite, up to ``max_retries`` times.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from openai import OpenAI

from ..llm_triage import OPENROUTER_BASE_URL
from . import prompts
from .models import DigestEntry, PaperDetails, PostDraft, Verdict, VerifyResult
from .verify import check_post

logger = logging.getLogger(__name__)

FILTER_BATCH_SIZE = 10
CONTRIBUTION_TYPES = ("verification", "validation", "uncertainty-quantification")

T = TypeVar("T")
Messages = List[Dict[str, str]]


class WriterError(RuntimeError):
    """The LLM never returned a parseable post."""


def make_client(api_key: str) -> OpenAI:
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


# ---------------------------------------------------------------------------
# Low-level call + JSON handling
# ---------------------------------------------------------------------------


def _call_llm(client: OpenAI, model: str, messages: Messages) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


def _load_json(raw: str) -> Any:
    """json.loads that also tolerates a ```json fence or stray text around the object."""
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise json.JSONDecodeError("no JSON object found", raw, 0)
    return json.loads(text[start : end + 1])


def _json_call(
    client: OpenAI, model: str, messages: Messages, parse: Callable[[Any], T], what: str
) -> Tuple[Optional[T], str]:
    """Call the LLM and parse the JSON; retry once with the strict JSON reminder.

    Returns (parsed value, "") or (None, failure reason).
    """
    reason = ""
    for attempt in range(2):
        attempt_messages = messages if attempt == 0 else prompts.with_json_retry_prefix(messages)
        try:
            raw = _call_llm(client, model, attempt_messages)
        except Exception as exc:  # noqa: BLE001 - one failed call must never kill the run
            reason = f"LLM call failed: {exc}"
            logger.warning("%s: LLM call failed (attempt %d): %s", what, attempt + 1, exc)
            continue
        try:
            return parse(_load_json(raw)), ""
        except (ValueError, KeyError, TypeError) as exc:
            reason = f"invalid JSON response: {exc}"
            logger.warning("%s: %s (attempt %d)", what, reason, attempt + 1)
    return None, reason


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


# ---------------------------------------------------------------------------
# KEEP/DROP filter
# ---------------------------------------------------------------------------


def _normalize_contribution(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    if text.startswith("verif"):
        return "verification"
    if text.startswith("valid"):
        return "validation"
    if text in {"uq", "u-q"} or "uncertainty" in text or "calibration" in text:
        return "uncertainty-quantification"
    return ""


def _parse_filter_results(data: Any, batch_len: int) -> Dict[int, Dict[str, Any]]:
    """Validate a filter response; return {batch index: result}, possibly partial.

    Like llm_triage._parse_batch_response, an index outside the batch makes
    the whole response invalid. An item with an unusable verdict is skipped,
    so its paper counts as missing.
    """
    results = data["results"]
    if not isinstance(results, list):
        raise ValueError("'results' is not a list")
    by_index: Dict[int, Dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("'results' items must be objects")
        index = _as_int(item.get("index"))
        if index is None or not 0 <= index < batch_len:
            raise ValueError(f"index {item.get('index')!r} is outside the batch")
        verdict = str(item.get("verdict", "")).strip().upper()
        if verdict not in {"KEEP", "DROP"}:
            continue
        by_index.setdefault(index, {**item, "verdict": verdict})
    return by_index


def _filter_batch(
    client: OpenAI, model: str, system: str, batch: List[DigestEntry], details: Dict[str, PaperDetails]
) -> Tuple[Dict[int, Dict[str, Any]], str]:
    """Run one filter batch with one retry; return (results by index, reason if incomplete)."""
    messages: Messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompts.build_filter_user_message(batch, details)},
    ]
    best: Dict[int, Dict[str, Any]] = {}
    reason = ""
    for attempt in range(2):
        attempt_messages = messages if attempt == 0 else prompts.with_json_retry_prefix(messages)
        try:
            raw = _call_llm(client, model, attempt_messages)
        except Exception as exc:  # noqa: BLE001 - one batch must never kill the run
            reason = f"LLM call failed: {exc}"
            logger.warning("Filter LLM call failed (attempt %d): %s", attempt + 1, exc)
            continue
        try:
            by_index = _parse_filter_results(_load_json(raw), len(batch))
        except (ValueError, KeyError, TypeError) as exc:
            reason = f"invalid JSON response: {exc}"
            logger.warning("Filter batch: %s (attempt %d)", reason, attempt + 1)
            continue
        for index, result in by_index.items():
            best.setdefault(index, result)
        if len(best) == len(batch):
            return best, ""
        missing = sorted(set(range(len(batch))) - set(best))
        reason = f"no verdict returned for batch indices {missing}"
        logger.warning("Filter batch: %s (attempt %d)", reason, attempt + 1)
    return best, reason


def filter_papers(
    client: OpenAI,
    model: str,
    entries: List[DigestEntry],
    details: Dict[str, PaperDetails],
    batch_size: int = FILTER_BATCH_SIZE,
) -> Dict[str, Verdict]:
    """Return one Verdict per entry URL (top papers and honorable mentions alike).

    Papers whose batch fails twice get keep=False with an "LLM filter failed: ..." rationale.
    """
    system = prompts.build_filter_system_prompt(prompts.load_rules())
    verdicts: Dict[str, Verdict] = {}

    for start in range(0, len(entries), batch_size):
        batch = entries[start : start + batch_size]
        by_index, reason = _filter_batch(client, model, system, batch, details)
        if reason:
            failed = [e.title for i, e in enumerate(batch) if i not in by_index]
            logger.warning("LLM filter failed for %d paper(s) (%s): %s", len(failed), reason, failed)

        for i, entry in enumerate(batch):
            result = by_index.get(i)
            if result is None:
                verdicts[entry.url] = Verdict(
                    url=entry.url, keep=False, rationale=f"LLM filter failed: {reason}"
                )
                continue
            keep = result["verdict"] == "KEEP"
            verdicts[entry.url] = Verdict(
                url=entry.url,
                keep=keep,
                rationale=" ".join(str(result.get("rationale", "")).split()),
                summary=str(result.get("summary", "") or "").strip() if keep else "",
                contribution_type=_normalize_contribution(result.get("contribution_type")) if keep else "",
            )

    kept = sum(1 for v in verdicts.values() if v.keep)
    logger.info("LLM filter: %d KEEP / %d DROP out of %d papers", kept, len(verdicts) - kept, len(verdicts))
    return verdicts


# ---------------------------------------------------------------------------
# Figure choice
# ---------------------------------------------------------------------------


def _parse_figure_choices(data: Any) -> List[Any]:
    choices = data["choices"]
    if not isinstance(choices, list):
        raise ValueError("'choices' is not a list")
    return choices


def choose_figures(client: OpenAI, model: str, captions_by_url: Dict[str, List[str]]) -> Dict[str, int]:
    """Pick the index of each paper's main overview/architecture/method diagram.

    One call for all papers. Returns {} on failure; out-of-range picks are dropped.
    """
    urls = [url for url, captions in captions_by_url.items() if captions]
    if not urls:
        return {}
    captions = [captions_by_url[url] for url in urls]
    messages: Messages = [
        {"role": "system", "content": prompts.FIGURE_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.build_figure_user_message(captions)},
    ]
    choices, reason = _json_call(client, model, messages, _parse_figure_choices, "Figure choice")
    if choices is None:
        logger.warning("Figure choice failed (%s); callers fall back to caption keywords", reason)
        return {}

    picked: Dict[str, int] = {}
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        paper = _as_int(choice.get("paper"))
        figure = _as_int(choice.get("figure"))
        if paper is None or figure is None or not 0 <= paper < len(urls):
            logger.warning("Dropping invalid figure choice %r", choice)
            continue
        if not 0 <= figure < len(captions[paper]):
            logger.warning(
                "Dropping out-of-range figure %d for %s (%d captions)", figure, urls[paper], len(captions[paper])
            )
            continue
        picked.setdefault(urls[paper], figure)
    return picked


# ---------------------------------------------------------------------------
# Post writing
# ---------------------------------------------------------------------------


def _parse_post(data: Any) -> Tuple[str, str, List[Dict[str, str]]]:
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    title, intro, papers = data["title"], data["intro"], data["papers"]
    if not isinstance(title, str) or not isinstance(intro, str):
        raise ValueError("'title' and 'intro' must be strings")
    if not isinstance(papers, list) or not all(isinstance(p, dict) for p in papers):
        raise ValueError("'papers' must be a list of objects")
    return title, intro, [{"url": str(p.get("url") or ""), "body": str(p.get("body") or "")} for p in papers]


def _clean_body(body: str, entry: DigestEntry) -> str:
    """Strip a lead-in that repeats the Python-built "**[Title](url)** — " prefix."""
    body = body.strip()
    body = re.sub(r"^\*\*\[[^\n]*?\]\([^)\s]*\)\*\*\s*[—–-]\s*", "", body)
    body = re.sub(r"^\*\*" + re.escape(entry.title) + r"\*\*\s*[—–:-]\s*", "", body)
    return body


def _to_post_draft(
    title: str, intro: str, papers: List[Dict[str, str]], keep_entries: List[DigestEntry]
) -> Tuple[PostDraft, List[str]]:
    """Map the LLM's papers back to KEEP entries by URL; report unusable entries as errors."""
    by_url = {e.url: e for e in keep_entries}
    bodies: Dict[str, str] = {}
    errors: List[str] = []
    for i, paper in enumerate(papers):
        url = paper["url"].strip()
        if not url:
            errors.append(f'papers[{i}] has no "url"; copy each paper URL exactly')
            continue
        entry = by_url.get(url)
        if entry is None:
            errors.append(f'papers[{i}] url "{url}" is not one of the KEEP paper URLs; copy each URL exactly')
            continue
        if url in bodies:
            errors.append(f'papers[{i}] repeats url "{url}"; give exactly one body per paper')
            continue
        bodies[url] = _clean_body(paper["body"], entry)
    clean_title = " ".join(title.strip().lstrip("#").split())
    return PostDraft(title=clean_title, intro=intro.strip(), bodies=bodies), errors


def write_post(
    client: OpenAI,
    model: str,
    keep_entries: List[DigestEntry],
    verdicts: Dict[str, Verdict],
    details: Dict[str, PaperDetails],
    all_entries: List[DigestEntry],
    style_example: Optional[str] = None,
    max_retries: int = 2,
) -> Tuple[PostDraft, VerifyResult]:
    """Write the post, verify it, and rewrite on hard errors up to ``max_retries`` times.

    Returns the last parseable attempt and its VerifyResult, even when errors
    remain. Raises WriterError only if the LLM never returned parseable JSON.
    """
    system = prompts.build_writer_system_prompt(prompts.load_rules(), style_example)
    base: Messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompts.build_writer_user_message(keep_entries, verdicts, details)},
    ]
    messages = base
    last: Optional[Tuple[PostDraft, VerifyResult]] = None
    reason = ""

    for attempt in range(max_retries + 1):
        parsed, reason = _json_call(client, model, messages, _parse_post, "Post writer")
        if parsed is None:
            logger.warning("Writer attempt %d returned no parseable post: %s", attempt + 1, reason)
            continue
        title, intro, papers = parsed
        post, mapping_errors = _to_post_draft(title, intro, papers, keep_entries)
        checked = check_post(post, keep_entries, all_entries, details)
        result = VerifyResult(errors=mapping_errors + checked.errors, warnings=checked.warnings)
        last = (post, result)
        if result.ok:
            logger.info("Post passed verification on attempt %d (%d warnings)", attempt + 1, len(result.warnings))
            return last
        logger.warning(
            "Writer attempt %d failed %d check(s): %s", attempt + 1, len(result.errors), result.errors
        )
        messages = base + [
            {"role": "assistant", "content": prompts.dump_post_json(title, intro, papers)},
            {"role": "user", "content": prompts.build_rewrite_message(result.errors)},
        ]

    if last is None:
        raise WriterError(f"LLM never returned a parseable post: {reason}")
    logger.warning("Post still fails %d check(s) after %d rewrite(s)", len(last[1].errors), max_retries)
    return last
