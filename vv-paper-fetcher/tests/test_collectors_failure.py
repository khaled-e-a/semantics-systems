from datetime import date
from unittest.mock import patch

import requests

from src.collectors.base import collect_all
from src.models import RunStats

WINDOW_START = date(2026, 8, 4)
WINDOW_END = date(2026, 8, 11)

CONFIG = {
    "sources": {
        "arxiv": {"enabled": True, "categories": ["cs.AI"], "max_results_per_call": 200, "max_total_results": 1000},
        "hf_papers": {"enabled": True},
        "openreview": {"enabled": True, "venues": []},
        "dblp": {"enabled": True, "venues": []},
    }
}


@patch("src.collectors.arxiv.requests.get", side_effect=requests.ConnectionError("boom"))
@patch("src.collectors.hf_papers.requests.get", side_effect=requests.ConnectionError("boom"))
def test_source_outage_is_isolated_and_does_not_raise(mock_hf_get, mock_arxiv_get):
    stats = RunStats()
    state = {}

    papers = collect_all(CONFIG, WINDOW_START, WINDOW_END, state, stats, only_sources=["arxiv", "hf_papers"])

    assert papers == []
    assert stats.collected_per_source["arxiv"] == 0
    assert stats.collected_per_source["hf_papers"] == 0
    assert "arxiv" in stats.source_errors
    assert "hf_papers" in stats.source_errors


@patch("src.collectors.hf_papers.requests.get", side_effect=requests.ConnectionError("boom"))
def test_one_source_failing_does_not_block_others(mock_hf_get):
    stats = RunStats()
    state = {}

    # openreview/dblp have no configured venues, so they trivially succeed with [] and no error.
    papers = collect_all(CONFIG, WINDOW_START, WINDOW_END, state, stats, only_sources=["hf_papers", "openreview", "dblp"])

    assert papers == []
    assert "hf_papers" in stats.source_errors
    assert "openreview" not in stats.source_errors
    assert "dblp" not in stats.source_errors
