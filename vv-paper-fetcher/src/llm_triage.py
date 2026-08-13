"""LLM triage via OpenRouter: batched relevance/summary/tag calls.

Title + abstract only (no full-PDF reading). Batches ~10-15 papers per call
with a structured JSON response to bound cost. Malformed JSON gets one
retry with a stricter follow-up; if that also fails, the batch is logged
and skipped rather than crashing the run.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from openai import OpenAI

from .models import Paper, RunStats, TriagedPaper

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = (
    "You are a research triage assistant for a weekly newsletter about verification, "
    "validation, uncertainty quantification, and evaluation methods for AI systems — "
    "specifically LLMs, AI agents, symbolic AI, scientific AI, and formal methods for AI. "
    "For each paper (title + abstract only), assess topical relevance and produce a short "
    "summary. Respond only with the specified JSON, no other text."
)


def _build_user_message(batch: List[Paper], max_abstract_chars: int, tag_vocabulary: List[str]) -> str:
    items = []
    for i, paper in enumerate(batch):
        abstract = paper.abstract[:max_abstract_chars]
        items.append(f"{i}. Title: {paper.title}\n   Abstract: {abstract}")
    papers_block = "\n".join(items)
    tags_block = ", ".join(tag_vocabulary)

    return (
        f"Papers:\n{papers_block}\n\n"
        f"Controlled tag vocabulary (use these where applicable, plus at most one "
        f"free-form tag per paper if needed): {tags_block}\n\n"
        'Respond with ONLY this JSON shape: {"results": [{"index": <int>, '
        '"relevant": <bool>, "relevance_score": <int 0-5>, "summary": "<1-2 sentences>", '
        '"tags": ["<tag>", ...]}, ...]} — one entry per paper, indices matching the list above.'
    )


def _call_llm(client: OpenAI, model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


def _parse_batch_response(raw: str, batch_len: int) -> List[Dict[str, Any]] | None:
    try:
        data = json.loads(raw)
        results = data["results"]
        if not isinstance(results, list):
            return None
        indices = {r["index"] for r in results}
        if not indices.issubset(set(range(batch_len))):
            return None
        return results
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def triage_papers(
    papers: List[Paper],
    api_key: str,
    model: str,
    llm_cfg: Dict[str, Any],
    stats: RunStats,
) -> List[TriagedPaper]:
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    batch_size = llm_cfg.get("batch_size", 12)
    max_abstract_chars = llm_cfg.get("max_abstract_chars", 1200)
    min_relevance_score = llm_cfg.get("min_relevance_score", 3)
    tag_vocabulary = llm_cfg.get("tag_vocabulary", [])

    triaged: List[TriagedPaper] = []

    for start in range(0, len(papers), batch_size):
        batch = papers[start : start + batch_size]
        user_msg = _build_user_message(batch, max_abstract_chars, tag_vocabulary)
        stats.llm_batches_total += 1

        results = None
        raw = ""
        for attempt in range(2):
            try:
                system = SYSTEM_PROMPT
                if attempt == 1:
                    user_msg = (
                        "Your previous response was not valid JSON matching the required "
                        "schema. Respond again with ONLY the JSON object.\n\n" + user_msg
                    )
                raw = _call_llm(client, model, system, user_msg)
                results = _parse_batch_response(raw, len(batch))
                if results is not None:
                    break
            except Exception as exc:  # noqa: BLE001 - one batch must never kill the run
                logger.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)

        if results is None:
            titles = [p.title for p in batch]
            logger.warning("Skipping LLM batch after failed parse; titles: %s", titles)
            stats.llm_batches_failed += 1
            continue

        by_index = {r["index"]: r for r in results}
        for i, paper in enumerate(batch):
            r = by_index.get(i)
            if r is None:
                continue
            relevance_score = int(r.get("relevance_score", 0))
            triaged.append(
                TriagedPaper(
                    paper=paper,
                    relevant=bool(r.get("relevant", False)) and relevance_score >= min_relevance_score,
                    relevance_score=relevance_score,
                    summary=str(r.get("summary", "")),
                    tags=list(r.get("tags", [])),
                )
            )

    stats.after_llm_triage = sum(1 for t in triaged if t.relevant)
    return triaged
