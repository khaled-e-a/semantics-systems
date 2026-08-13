import json
from datetime import date
from unittest.mock import MagicMock, patch

from src.llm_triage import triage_papers
from src.models import Paper, RunStats

LLM_CFG = {
    "batch_size": 12,
    "max_abstract_chars": 1200,
    "min_relevance_score": 3,
    "tag_vocabulary": ["verification", "evals"],
}


def _paper(title="A paper"):
    return Paper(
        title=title,
        authors=["Someone"],
        abstract="an abstract",
        url="https://example.com",
        venue="cs.AI",
        source="arxiv",
        published_date=date(2026, 8, 4),
        arxiv_id="2401.00001",
    )


def _mock_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    return resp


@patch("src.llm_triage.OpenAI")
def test_valid_json_response_produces_triaged_papers(mock_openai_cls):
    valid_json = json.dumps(
        {"results": [{"index": 0, "relevant": True, "relevance_score": 5, "summary": "sum", "tags": ["evals"]}]}
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_response(valid_json)
    mock_openai_cls.return_value = mock_client

    stats = RunStats()
    result = triage_papers([_paper()], "fake-key", "fake-model", LLM_CFG, stats)

    assert len(result) == 1
    assert result[0].relevant is True
    assert result[0].relevance_score == 5
    assert stats.llm_batches_failed == 0


@patch("src.llm_triage.OpenAI")
def test_malformed_json_on_both_attempts_skips_batch_without_raising(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_response("not json at all")
    mock_openai_cls.return_value = mock_client

    stats = RunStats()
    result = triage_papers([_paper()], "fake-key", "fake-model", LLM_CFG, stats)

    assert result == []
    assert stats.llm_batches_failed == 1
    assert stats.llm_batches_total == 1
    assert mock_client.chat.completions.create.call_count == 2  # one retry


@patch("src.llm_triage.OpenAI")
def test_malformed_then_valid_on_retry_succeeds(mock_openai_cls):
    valid_json = json.dumps(
        {"results": [{"index": 0, "relevant": True, "relevance_score": 4, "summary": "sum", "tags": []}]}
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [_mock_response("garbage"), _mock_response(valid_json)]
    mock_openai_cls.return_value = mock_client

    stats = RunStats()
    result = triage_papers([_paper()], "fake-key", "fake-model", LLM_CFG, stats)

    assert len(result) == 1
    assert stats.llm_batches_failed == 0


@patch("src.llm_triage.OpenAI")
def test_below_threshold_relevance_score_marked_not_relevant(mock_openai_cls):
    valid_json = json.dumps(
        {"results": [{"index": 0, "relevant": True, "relevance_score": 1, "summary": "sum", "tags": []}]}
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_response(valid_json)
    mock_openai_cls.return_value = mock_client

    stats = RunStats()
    result = triage_papers([_paper()], "fake-key", "fake-model", LLM_CFG, stats)

    assert result[0].relevant is False
