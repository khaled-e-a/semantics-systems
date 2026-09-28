from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import requests

from src.substack_post import paper_details
from src.substack_post.models import DigestEntry, PaperDetails
from src.substack_post.paper_details import (
    API_URL,
    OAI_URL,
    fetch_details,
    fetch_licenses,
    is_open_license,
    is_valid_arxiv_id,
    parse_api_feed,
    parse_oai_license,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "substack"
API_FIXTURE = (FIXTURES / "arxiv_api_batch.xml").read_bytes()  # asked 28614v1,29935v1,30489v1,11115
OAI_FIXTURE = (FIXTURES / "oai_pmh_2609.28614.xml").read_bytes()


class FakeResponse:
    def __init__(self, status_code: int = 200, content: bytes = b"") -> None:
        self.status_code = status_code
        self.content = content
        self.headers: Dict[str, str] = {}

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class FakeSession:
    """Answers every request with `handler(url, params)`; records the calls."""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: List[Dict[str, Any]] = []

    def get(self, url: str, params: Optional[Dict[str, str]] = None, **kwargs) -> FakeResponse:
        self.calls.append({"url": url, "params": dict(params or {}), **kwargs})
        result = self.handler(url, params or {})
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture(autouse=True)
def no_throttle(monkeypatch):
    monkeypatch.setattr(paper_details._api_throttle, "min_interval_s", 0)


def _entry(rank: int, url: str, arxiv_id: Optional[str], version: Optional[str]) -> DigestEntry:
    return DigestEntry(rank=rank, section="top", title=f"Paper {rank}", url=url, arxiv_id=arxiv_id, version=version)


def test_parse_api_feed_fixture():
    by_id = parse_api_feed(API_FIXTURE)
    assert set(by_id) == {"2609.28614", "2609.29935", "2609.30489", "2609.11115"}

    reward = by_id["2609.28614"][0]
    assert reward.version == "v1"
    assert reward.title == "Reward Hacking Challenges Oversight of Autonomous Research Agents"
    assert len(reward.authors) == 15 and reward.authors[0] == "Yue Huang"
    assert reward.abstract.startswith("Autonomous research agents can design experiments")
    assert "\n" not in reward.abstract and "  " not in reward.abstract
    assert reward.license_url is None

    assert by_id["2609.11115"][0].version == "v3"  # bare id → latest version
    assert len(by_id["2609.30489"][0].authors) == 63  # FULL author list, not the digest's 3


def test_fetch_details_matches_out_of_order_entries():
    entries = [
        _entry(1, "http://arxiv.org/abs/2609.30489v1", "2609.30489", "v1"),
        _entry(2, "http://arxiv.org/abs/2609.29935v1", "2609.29935", "v1"),
        _entry(3, "http://arxiv.org/abs/2609.28614v1", "2609.28614", "v1"),
        _entry(4, "https://huggingface.co/papers/2609.11115", "2609.11115", None),
        _entry(5, "https://openreview.net/forum?id=x", None, None),
        _entry(6, "http://arxiv.org/abs/2609.99999v1", "2609.99999", "v1"),  # well-formed, unknown
        _entry(7, "http://arxiv.org/abs/bogus", "26.09", None),  # malformed → never sent
    ]
    session = FakeSession(lambda url, params: FakeResponse(200, API_FIXTURE))

    details = fetch_details(entries, session=session)

    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == API_URL and API_URL.startswith("https://")
    ids = call["params"]["id_list"].split(",")
    assert ids == ["2609.30489v1", "2609.29935v1", "2609.28614v1", "2609.11115", "2609.99999v1"]
    assert call["params"]["max_results"] == "5"
    assert call["timeout"]

    assert set(details) == {e.url for e in entries[:4]}
    assert details[entries[0].url].arxiv_id == "2609.30489"
    assert details[entries[1].url].title.startswith("Robust Detection of LLM-Generated Text")
    assert details[entries[2].url].authors[0] == "Yue Huang"
    assert details[entries[3].url].version == "v3"


def test_fetch_details_batches_of_fifty_with_max_results():
    entries = [
        _entry(i, f"http://arxiv.org/abs/2609.{10000 + i}v1", f"2609.{10000 + i}", "v1") for i in range(1, 121)
    ]
    session = FakeSession(lambda url, params: FakeResponse(200, API_FIXTURE))

    fetch_details(entries, session=session)

    sizes = [len(c["params"]["id_list"].split(",")) for c in session.calls]
    assert sizes == [50, 50, 20]
    assert [c["params"]["max_results"] for c in session.calls] == ["50", "50", "20"]


def test_fetch_details_waits_between_batches(monkeypatch):
    sleeps: List[float] = []
    monkeypatch.setattr(paper_details._api_throttle, "min_interval_s", 3.0)
    monkeypatch.setattr(paper_details.time, "sleep", lambda s: sleeps.append(s))
    entries = [_entry(i, f"u{i}", f"2609.{10000 + i}", None) for i in range(1, 61)]

    fetch_details(entries, session=FakeSession(lambda url, params: FakeResponse(200, API_FIXTURE)))

    assert sleeps and all(0 < s <= 3.0 for s in sleeps)


def test_failed_batch_is_skipped_without_raising():
    entries = [_entry(1, "http://arxiv.org/abs/2609.28614v1", "2609.28614", "v1")]
    assert fetch_details(entries, session=FakeSession(lambda u, p: FakeResponse(400, b"bad"))) == {}
    assert fetch_details(entries, session=FakeSession(lambda u, p: requests.ConnectionError("down"))) == {}


def test_fetch_details_without_arxiv_ids_makes_no_call():
    session = FakeSession(lambda u, p: FakeResponse(200, API_FIXTURE))
    assert fetch_details([_entry(1, "https://openreview.net/forum?id=x", None, None)], session=session) == {}
    assert session.calls == []


def test_parse_oai_license_fixture():
    assert parse_oai_license(OAI_FIXTURE) == "http://creativecommons.org/licenses/by/4.0/"
    without = OAI_FIXTURE.replace(b"<license>http://creativecommons.org/licenses/by/4.0/</license>", b"")
    assert parse_oai_license(without) is None


def _details(arxiv_id: str) -> PaperDetails:
    return PaperDetails(arxiv_id=arxiv_id, version="v1", title="T", abstract="A", authors=["X"])


def test_fetch_licenses_fills_in_place():
    shared = _details("2609.28614")
    details = {
        "http://arxiv.org/abs/2609.28614v1": shared,
        "https://huggingface.co/papers/2609.28614": shared,  # same paper under two URLs
        "http://arxiv.org/abs/2609.29935v1": _details("2609.29935"),
    }
    session = FakeSession(lambda url, params: FakeResponse(200, OAI_FIXTURE))

    fetch_licenses(
        details,
        ["http://arxiv.org/abs/2609.28614v1", "https://huggingface.co/papers/2609.28614", "missing-url"],
        session=session,
    )

    assert len(session.calls) == 1  # one lookup per arXiv id; unknown urls ignored
    call = session.calls[0]
    assert call["url"] == OAI_URL
    assert call["params"] == {
        "verb": "GetRecord",
        "identifier": "oai:arXiv.org:2609.28614",
        "metadataPrefix": "arXiv",
    }
    assert shared.license_url == "http://creativecommons.org/licenses/by/4.0/"
    assert details["http://arxiv.org/abs/2609.29935v1"].license_url is None  # not a KEEP url


def test_fetch_licenses_failure_leaves_none():
    details = {"u": _details("2609.28614")}
    fetch_licenses(details, ["u"], session=FakeSession(lambda url, params: requests.Timeout("slow")))
    assert details["u"].license_url is None
    fetch_licenses(details, ["u"], session=FakeSession(lambda url, params: FakeResponse(200, b"<not xml")))
    assert details["u"].license_url is None


@pytest.mark.parametrize(
    "license_url, expected",
    [
        ("http://creativecommons.org/licenses/by/4.0/", True),
        ("https://creativecommons.org/licenses/by/3.0/", True),
        ("http://creativecommons.org/licenses/by/4.0", True),
        ("http://creativecommons.org/licenses/by-sa/4.0/", True),
        ("http://creativecommons.org/publicdomain/zero/1.0/", True),
        ("http://creativecommons.org/licenses/by-nc/4.0/", False),
        ("http://creativecommons.org/licenses/by-nc-sa/4.0/", False),
        ("http://creativecommons.org/licenses/by-nc-nd/4.0/", False),
        ("http://creativecommons.org/licenses/by-nd/4.0/", False),
        ("http://arxiv.org/licenses/nonexclusive-distrib/1.0/", False),
        ("http://creativecommons.org/licenses/publicdomain/", False),
        ("", False),
        (None, False),
    ],
)
def test_is_open_license(license_url, expected):
    assert is_open_license(license_url) is expected


@pytest.mark.parametrize(
    "arxiv_id, expected",
    [
        ("2609.28614", True),
        ("2609.28614v1", True),
        ("1501.0001", True),
        ("2609.28614v", False),
        ("2609.286", False),
        ("26.09", False),
        ("hep-th/9901001", False),
        ("2609.28614 ", False),
        ("2609.28614,2609.29935", False),
        ("", False),
    ],
)
def test_is_valid_arxiv_id(arxiv_id, expected):
    assert is_valid_arxiv_id(arxiv_id) is expected
