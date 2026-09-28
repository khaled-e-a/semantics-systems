"""writer.py + prompts.py with a mocked OpenAI client (no network)."""
import json
from typing import List, Union
from unittest.mock import MagicMock, patch

import pytest

from src.llm_triage import OPENROUTER_BASE_URL
from src.substack_post import prompts
from src.substack_post.models import DigestEntry, PaperDetails, Verdict
from src.substack_post.writer import (
    WriterError,
    choose_figures,
    filter_papers,
    make_client,
    write_post,
)


def _resp(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    return resp


def _client(responses: List[Union[str, Exception]]):
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        r if isinstance(r, Exception) else _resp(r) for r in responses
    ]
    return client


def _messages(client, call_index: int):
    return client.chat.completions.create.call_args_list[call_index].kwargs["messages"]


def _entry(i: int, section: str = "top") -> DigestEntry:
    return DigestEntry(
        rank=i + 1,
        section=section,
        title=f"Paper {i}",
        url=f"http://arxiv.org/abs/2609.{10000 + i}v1",
        arxiv_id=f"2609.{10000 + i}",
        version="v1",
        digest_summary=f"one-liner {i}",
        tags=["evals"],
    )


def _filter_json(indices, keep=()):
    return json.dumps(
        {
            "results": [
                {
                    "index": i,
                    "verdict": "KEEP" if i in keep else "DROP",
                    "rationale": f"reason {i}",
                    "summary": f"summary {i}" if i in keep else "",
                    "contribution_type": "Uncertainty Quantification" if i in keep else "",
                }
                for i in indices
            ]
        }
    )


def test_make_client_uses_openrouter():
    with patch("src.substack_post.writer.OpenAI") as openai_cls:
        make_client("key")
    openai_cls.assert_called_once_with(base_url=OPENROUTER_BASE_URL, api_key="key")


# --- filter ----------------------------------------------------------------------------


def test_filter_batches_and_returns_one_verdict_per_entry():
    entries = [_entry(i) for i in range(23)]
    details = {entries[0].url: PaperDetails("2609.10000", "v1", "Paper 0", "Real abstract zero.", ["A B"])}
    client = _client(
        [_filter_json(range(10), keep={0}), _filter_json(range(10), keep={3}), _filter_json(range(3))]
    )

    verdicts = filter_papers(client, "m", entries, details)

    assert client.chat.completions.create.call_count == 3
    assert set(verdicts) == {e.url for e in entries}
    kept = sorted(url for url, v in verdicts.items() if v.keep)
    assert kept == sorted([entries[0].url, entries[13].url])
    v0 = verdicts[entries[0].url]
    assert v0.summary == "summary 0" and v0.contribution_type == "uncertainty-quantification"
    dropped = verdicts[entries[1].url]
    assert dropped.summary == "" and dropped.contribution_type == "" and dropped.rationale == "reason 1"

    first_user = _messages(client, 0)[1]["content"]
    assert "Real abstract zero." in first_user
    assert prompts.ABSTRACT_UNAVAILABLE in first_user
    assert "Paper 9" in first_user and "Paper 10" not in first_user
    assert "Paper 20" in _messages(client, 2)[1]["content"]
    call = client.chat.completions.create.call_args_list[0].kwargs
    assert call["response_format"] == {"type": "json_object"}
    assert call["model"] == "m"


def test_filter_system_prompt_uses_section_2_of_the_rules_doc():
    client = _client([_filter_json(range(1))])
    filter_papers(client, "m", [_entry(0)], {})
    system = _messages(client, 0)[0]["content"]
    assert "Drop everything else" in system
    assert "Do not trust the tags" in system
    assert "How to do this filtering in practice" not in system  # agent/WebFetch mechanics removed
    assert "25–35%" in system


def test_filter_details_can_be_keyed_by_arxiv_id():
    entry = _entry(0)
    details = {"2609.10000": PaperDetails("2609.10000", "v1", "Paper 0", "Abstract by id.", [])}
    client = _client([_filter_json(range(1))])
    filter_papers(client, "m", [entry], details)
    assert "Abstract by id." in _messages(client, 0)[1]["content"]


def test_filter_out_of_range_index_triggers_retry():
    entries = [_entry(i) for i in range(2)]
    bad = json.dumps({"results": [{"index": 5, "verdict": "KEEP", "rationale": "x"}]})
    client = _client([bad, _filter_json(range(2), keep={1})])

    verdicts = filter_papers(client, "m", entries, {})

    assert client.chat.completions.create.call_count == 2
    assert _messages(client, 1)[1]["content"].startswith(prompts.JSON_RETRY_PREFIX)
    assert verdicts[entries[1].url].keep is True


def test_filter_failed_batch_becomes_drop_without_affecting_other_batches():
    entries = [_entry(i) for i in range(12)]
    client = _client(["not json", RuntimeError("boom"), _filter_json(range(2), keep={0, 1})])

    verdicts = filter_papers(client, "m", entries, {})

    assert len(verdicts) == 12
    for entry in entries[:10]:
        v = verdicts[entry.url]
        assert v.keep is False
        assert v.rationale.startswith("LLM filter failed:")
    assert verdicts[entries[10].url].keep and verdicts[entries[11].url].keep


def test_filter_partial_response_is_completed_by_the_retry():
    entries = [_entry(i) for i in range(3)]
    client = _client([_filter_json([0, 2], keep={0}), _filter_json([1], keep={1})])

    verdicts = filter_papers(client, "m", entries, {})

    assert client.chat.completions.create.call_count == 2
    assert [verdicts[e.url].keep for e in entries] == [True, True, False]


def test_filter_still_missing_after_retry_becomes_drop():
    entries = [_entry(i) for i in range(2)]
    client = _client([_filter_json([0], keep={0}), _filter_json([0], keep={0})])

    verdicts = filter_papers(client, "m", entries, {})

    assert verdicts[entries[0].url].keep is True
    assert verdicts[entries[1].url].keep is False
    assert verdicts[entries[1].url].rationale.startswith("LLM filter failed: no verdict returned")


# --- figures ---------------------------------------------------------------------------


def test_choose_figures_drops_out_of_range_choices():
    captions = {
        "u0": ["Figure 1: results", "Figure 2: overview of the pipeline"],
        "u1": ["Figure 1: architecture"],
        "u2": ["Figure 1: a", "Figure 2: b"],
        "u3": [],  # no captions: not sent
    }
    reply = json.dumps(
        {"choices": [{"paper": 0, "figure": 1}, {"paper": 1, "figure": 3}, {"paper": 2, "figure": -1},
                     {"paper": 7, "figure": 0}, {"paper": True, "figure": 0}]}
    )
    client = _client([reply])

    assert choose_figures(client, "m", captions) == {"u0": 1}
    assert client.chat.completions.create.call_count == 1
    user = _messages(client, 0)[1]["content"]
    assert "[1] Figure 2: overview of the pipeline" in user
    assert "Paper 3" not in user


def test_choose_figures_returns_empty_dict_on_failure():
    client = _client(["garbage", '{"picks": []}'])
    assert choose_figures(client, "m", {"u0": ["Figure 1: x"]}) == {}
    assert client.chat.completions.create.call_count == 2


def test_choose_figures_skips_call_when_there_are_no_captions():
    client = _client([])
    assert choose_figures(client, "m", {"u0": []}) == {}
    assert choose_figures(client, "m", {}) == {}
    client.chat.completions.create.assert_not_called()


# --- write_post ------------------------------------------------------------------------

KEEP = [_entry(0), _entry(3)]
ALL = [_entry(i) for i in range(5)]
VERDICTS = {
    KEEP[0].url: Verdict(KEEP[0].url, True, "r", "Summary zero.", "verification"),
    KEEP[1].url: Verdict(KEEP[1].url, True, "r", "Summary three.", "uncertainty-quantification"),
}
DETAILS = {
    KEEP[0].url: PaperDetails("2609.10000", "v1", "Paper 0", "Abstract zero.", ["Grace Hopper", "Alan Turing"]),
}

FILLER = (
    "The tool reads each answer and splits the answer into short claims. "
    "A second model checks every claim against the source text. "
    "The checker marks each claim as supported, contradicted, or unknown. "
    "On a test set of 500 answers, the checker found 91 percent of the planted errors. "
    "A simple baseline found 64 percent of the same errors. "
    "The checker also runs fast enough to check every answer before a person reads it. "
)


def _body(kind: str) -> str:
    return FILLER + f"The {kind} contribution is a check that a reader can repeat on new answers."


def _post_json(title="Two Ways to Check AI Answers", bodies=None, urls=None):
    bodies = bodies or [_body("verification"), _body("uncertainty-quantification")]
    urls = urls or [KEEP[0].url, KEEP[1].url]
    return json.dumps(
        {
            "title": title,
            "intro": "One paper checks claims against sources. Another paper measures confidence.",
            "papers": [{"url": u, "body": b} for u, b in zip(urls, bodies)],
        }
    )


def test_write_post_passes_first_time():
    client = _client([_post_json()])
    post, result = write_post(client, "m", KEEP, VERDICTS, DETAILS, ALL)
    assert result.ok, result.errors
    assert post.title == "Two Ways to Check AI Answers"
    assert set(post.bodies) == {KEEP[0].url, KEEP[1].url}
    assert client.chat.completions.create.call_count == 1


def test_write_post_prompts_include_rules_doc_style_example_and_papers():
    client = _client([_post_json()])
    write_post(client, "m", KEEP, VERDICTS, DETAILS, ALL, style_example="# 15 Ways\n\nAPPROVED-STYLE-MARKER")
    system, user = _messages(client, 0)
    rules = prompts.load_rules()
    assert rules.strip() in system["content"]
    assert "APPROVED-STYLE-MARKER" in system["content"]
    assert "previously approved post" in system["content"]
    assert "do not copy its content" in system["content"]
    assert "Never mention author names" in system["content"]
    assert "Summary zero." in user["content"] and "Abstract zero." in user["content"]
    assert user["content"].index(KEEP[0].url) < user["content"].index(KEEP[1].url)
    assert "Grace Hopper" not in system["content"] + user["content"]


def test_write_post_without_style_example():
    client = _client([_post_json()])
    write_post(client, "m", KEEP, VERDICTS, DETAILS, ALL)
    assert "previously approved post" not in _messages(client, 0)[0]["content"]


def test_write_post_rewrites_after_verify_errors():
    bad = _post_json(bodies=[_body("verification") + " Grace Hopper leads it.", _body("uncertainty-quantification")])
    client = _client([bad, _post_json()])

    post, result = write_post(client, "m", KEEP, VERDICTS, DETAILS, ALL)

    assert result.ok, result.errors
    assert "Grace Hopper" not in post.bodies[KEEP[0].url]
    assert client.chat.completions.create.call_count == 2
    rewrite = _messages(client, 1)
    assert rewrite[2]["role"] == "assistant" and "Grace Hopper leads it." in rewrite[2]["content"]
    assert rewrite[3]["role"] == "user" and '"Grace Hopper"' in rewrite[3]["content"]
    assert "failed these automated checks" in rewrite[3]["content"]


def test_write_post_returns_last_attempt_when_errors_persist():
    responses = [_post_json(title=f"Ways to Check, draft {c}") for c in "ABC"]
    client = _client(responses)

    post, result = write_post(client, "m", KEEP, VERDICTS, DETAILS, ALL, max_retries=2)

    assert client.chat.completions.create.call_count == 3
    assert not result.ok
    assert post.title == "Ways to Check, draft C"
    assert any("title must state the number of papers" in e for e in result.errors), result.errors


def test_write_post_unknown_url_is_an_error_that_triggers_rewrite():
    wrong = _post_json(urls=[KEEP[0].url.replace("v1", "v2"), KEEP[1].url + "  "])
    client = _client([wrong, _post_json()])

    post, result = write_post(client, "m", KEEP, VERDICTS, DETAILS, ALL)

    assert result.ok
    assert client.chat.completions.create.call_count == 2
    rewrite_request = _messages(client, 1)[3]["content"]
    assert "is not one of the KEEP paper URLs" in rewrite_request


def test_write_post_tolerates_trailing_whitespace_in_urls_and_strips_link_lead_in():
    lead_in = f"**[Paper 0]({KEEP[0].url})** — "
    reply = _post_json(urls=[KEEP[0].url + " ", KEEP[1].url], bodies=[lead_in + _body("verification"),
                                                                    _body("uncertainty-quantification")])
    client = _client([reply])
    post, result = write_post(client, "m", KEEP, VERDICTS, DETAILS, ALL)
    assert result.ok, result.errors
    assert post.bodies[KEEP[0].url].startswith("The tool reads")


def test_write_post_retries_bad_json_then_succeeds():
    client = _client(["```json\nnot really json\n```", _post_json()])
    post, result = write_post(client, "m", KEEP, VERDICTS, DETAILS, ALL)
    assert result.ok
    assert _messages(client, 1)[1]["content"].startswith(prompts.JSON_RETRY_PREFIX)


def test_write_post_accepts_fenced_json():
    client = _client(["Here you go:\n```json\n" + _post_json() + "\n```"])
    _, result = write_post(client, "m", KEEP, VERDICTS, DETAILS, ALL)
    assert result.ok


def test_write_post_raises_when_never_parseable():
    client = _client(["nope"] * 6)
    with pytest.raises(WriterError):
        write_post(client, "m", KEEP, VERDICTS, DETAILS, ALL, max_retries=2)
    assert client.chat.completions.create.call_count == 6


# --- prompts helpers -------------------------------------------------------------------


def test_load_style_example_picks_newest_and_honours_exclude_date(tmp_path):
    (tmp_path / "2026-08-31-substack-post-ste100.md").write_text("old", encoding="utf-8")
    (tmp_path / "2026-09-07-substack-post-ste100.md").write_text("newer", encoding="utf-8")
    (tmp_path / "2026-09-28-substack-post-ste100.md").write_text("newest", encoding="utf-8")
    (tmp_path / "2026-10-05-substack-post.md").write_text("not ste100", encoding="utf-8")
    (tmp_path / "2026-10-05.md").write_text("digest", encoding="utf-8")

    assert prompts.load_style_example(tmp_path) == "newest"
    assert prompts.load_style_example(tmp_path, exclude_date="2026-09-28") == "newer"


def test_load_style_example_returns_none_when_missing(tmp_path):
    assert prompts.load_style_example(tmp_path) is None
    (tmp_path / "2026-09-28-substack-post-ste100.md").write_text("only", encoding="utf-8")
    assert prompts.load_style_example(tmp_path, exclude_date="2026-09-28") is None


def test_load_style_example_finds_committed_posts():
    from src.config import PROJECT_ROOT

    example = prompts.load_style_example(PROJECT_ROOT / "reports", exclude_date="2099-01-01")
    assert example is not None and example.startswith("# ")


def test_extract_section_by_number_and_keyword():
    doc = "# Title\n\n## 1. Intro\n\none\n\n## 2. Filter things\n\ntwo\n\n### How in practice\n\nx\n\n## 3. Next\n\nthree"
    assert prompts.extract_section(doc, 2) == "## 2. Filter things\n\ntwo\n\n### How in practice\n\nx"
    assert prompts.extract_section(doc.replace("## 2.", "## Two."), 2, "filter").startswith("## Two. Filter")
    assert prompts.extract_section(doc, 9) is None
