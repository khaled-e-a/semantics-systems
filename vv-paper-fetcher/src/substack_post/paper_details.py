"""Fetch real arXiv metadata (abstract, full author list, license) for digest entries.

The digest only carries a 1-2 sentence LLM summary and the first three
authors, so the writer and the verify step need the real abstract and the
FULL author list from the arXiv export API:

- https only (plain http answers 301 with an empty body);
- `max_results` is always set (the default of 10 silently truncates);
- entries come back in arbitrary order, so they are matched by `<id>`;
- one malformed id turns the whole batch into HTTP 400, so ids are
  validated first; a well-formed but unknown id simply has no entry.

Licenses come from OAI-PMH (`<license>` in the arXiv metadata format) and
are fetched for KEEP papers only. arXiv asks for >= 3 s between calls to
both services.
"""
from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Tuple

import feedparser
import requests

from .models import DigestEntry, PaperDetails

logger = logging.getLogger(__name__)

API_URL = "https://export.arxiv.org/api/query"
OAI_URL = "https://oaipmh.arxiv.org/oai"
USER_AGENT = "vv-paper-fetcher/1.0 (weekly digest)"
TIMEOUT_S = 30
BATCH_SIZE = 50
API_DELAY_S = 3.0  # arXiv's guidance between successive export API / OAI-PMH calls

NS_OAI_ARXIV = "{http://arxiv.org/OAI/arXiv/}"

# New-style ids only: 2609.28614 or 2609.28614v2.
ARXIV_ID_RE = re.compile(r"^(?P<id>\d{4}\.\d{4,5})(?P<ver>v\d+)?$")
# <id>http://arxiv.org/abs/2609.28614v1</id> in the Atom feed.
_ABS_ID_RE = re.compile(r"arxiv\.org/abs/(?P<id>\d{4}\.\d{4,5})(?P<ver>v\d+)?\s*$")

_OPEN_LICENSE_RE = re.compile(
    r"^https?://(?:www\.)?creativecommons\.org/"
    r"(?:licenses/(?:by|by-sa)/\d+(?:\.\d+)*|publicdomain/zero/\d+(?:\.\d+)*)"
    r"(?:/(?:legalcode(?:\.[a-z-]+)?)?)?/?$",
    re.IGNORECASE,
)


class Throttle:
    """Keep at least `min_interval_s` seconds between successive calls to wait()."""

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = min_interval_s
        self._last: Optional[float] = None

    def wait(self) -> None:
        if self._last is not None:
            remaining = self.min_interval_s - (time.monotonic() - self._last)
            if remaining > 0:
                time.sleep(remaining)
        self._last = time.monotonic()


# Module-level so the spacing holds across calls within one process.
_api_throttle = Throttle(API_DELAY_S)


def is_valid_arxiv_id(arxiv_id: str) -> bool:
    """True for a new-style arXiv id with an optional version (2609.28614, 2609.28614v1)."""
    return bool(arxiv_id) and ARXIV_ID_RE.match(arxiv_id) is not None


def is_open_license(license_url: Optional[str]) -> bool:
    """True for CC BY, CC BY-SA (any version) and CC0; False for NC/ND, arXiv's license, None."""
    if not license_url:
        return False
    return _OPEN_LICENSE_RE.match(license_url.strip()) is not None


def new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _collapse(text: str) -> str:
    return " ".join((text or "").split())


def parse_api_feed(content: bytes) -> Dict[str, List[PaperDetails]]:
    """Parse an export API Atom feed into {bare_id: [PaperDetails, ...]}.

    Entries without a recognizable abs id (e.g. arXiv's error entries) are skipped.
    """
    feed = feedparser.parse(content)
    by_id: Dict[str, List[PaperDetails]] = {}
    for entry in feed.entries:
        m = _ABS_ID_RE.search(entry.get("id", ""))
        if not m:
            logger.debug("Skipping arXiv feed entry without an abs id: %r", entry.get("id"))
            continue
        details = PaperDetails(
            arxiv_id=m.group("id"),
            version=m.group("ver"),
            title=_collapse(entry.get("title", "")),
            abstract=_collapse(entry.get("summary", "")),
            authors=[_collapse(a.get("name", "")) for a in entry.get("authors", []) if a.get("name")],
        )
        by_id.setdefault(details.arxiv_id, []).append(details)
    return by_id


def _pick(candidates: List[PaperDetails], version: Optional[str]) -> Optional[PaperDetails]:
    if not candidates:
        return None
    if version:
        for d in candidates:
            if d.version == version:
                return d
    return candidates[0]


def fetch_details(
    entries: List[DigestEntry], session: Optional[requests.Session] = None
) -> Dict[str, PaperDetails]:
    """Fetch arXiv metadata for entries that have an arXiv id; keyed by entry.url.

    Entries without an arxiv_id, with a malformed id, in a failed batch, or not
    returned by arXiv are simply absent from the result.
    """
    session = session or new_session()

    # query id ("2609.28614v1" or bare "2609.11115") -> entries asking for it
    wanted: Dict[str, List[DigestEntry]] = {}
    for entry in entries:
        if not entry.arxiv_id:
            continue
        query_id = f"{entry.arxiv_id}{entry.version or ''}"
        if not is_valid_arxiv_id(query_id):
            logger.warning("Skipping malformed arXiv id %r for %s", query_id, entry.url)
            continue
        wanted.setdefault(query_id, []).append(entry)

    query_ids = list(wanted)
    results: Dict[str, PaperDetails] = {}
    for start in range(0, len(query_ids), BATCH_SIZE):
        batch = query_ids[start : start + BATCH_SIZE]
        params = {"id_list": ",".join(batch), "max_results": str(len(batch))}
        try:
            _api_throttle.wait()
            resp = session.get(API_URL, params=params, timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            by_id = parse_api_feed(resp.content)
        except Exception as exc:  # noqa: BLE001 - one failed batch must never kill the run
            logger.warning("arXiv API batch of %d ids failed: %s", len(batch), exc)
            continue

        for query_id in batch:
            m = ARXIV_ID_RE.match(query_id)
            details = _pick(by_id.get(m.group("id"), []), m.group("ver")) if m else None
            if details is None:
                logger.warning("arXiv returned no entry for %s", query_id)
                continue
            for entry in wanted[query_id]:
                results[entry.url] = details

    logger.info("Fetched arXiv details for %d/%d entries", len(results), len(entries))
    return results


def parse_oai_license(content: bytes) -> Optional[str]:
    """Return the <license> URL from an OAI-PMH GetRecord (arXiv format) response, or None."""
    root = ET.fromstring(content)
    node = root.find(f".//{NS_OAI_ARXIV}license")
    if node is None or not (node.text or "").strip():
        return None
    return node.text.strip()


def fetch_licenses(
    details: Dict[str, PaperDetails], urls: Iterable[str], session: Optional[requests.Session] = None
) -> None:
    """Fill details[url].license_url in place for the given URLs (KEEP papers only).

    A failed lookup leaves license_url as None (treated as not openly licensed).
    """
    session = session or new_session()
    cache: Dict[str, Tuple[bool, Optional[str]]] = {}  # arxiv_id -> (fetched, license)

    for url in urls:
        paper = details.get(url)
        if paper is None:
            continue
        if paper.arxiv_id not in cache:
            cache[paper.arxiv_id] = (False, None)
            params = {
                "verb": "GetRecord",
                "identifier": f"oai:arXiv.org:{paper.arxiv_id}",
                "metadataPrefix": "arXiv",
            }
            try:
                _api_throttle.wait()
                resp = session.get(OAI_URL, params=params, timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT})
                resp.raise_for_status()
                cache[paper.arxiv_id] = (True, parse_oai_license(resp.content))
            except Exception as exc:  # noqa: BLE001 - a missing license is not fatal
                logger.warning("OAI-PMH license lookup failed for %s: %s", paper.arxiv_id, exc)
        fetched, license_url = cache[paper.arxiv_id]
        if fetched:
            paper.license_url = license_url
