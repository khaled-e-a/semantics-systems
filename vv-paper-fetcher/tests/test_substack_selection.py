from types import SimpleNamespace

import pytest

from src.substack_post.models import DigestEntry
from src.substack_post.selection import ranking_key, select_papers, tier_of


def _entry(rank, h_index=None):
    return DigestEntry(
        rank=rank, section="top", title=f"Paper {rank}", url=f"http://arxiv.org/abs/2609.{rank:05d}v1", max_h_index=h_index
    )


def _aff(tier):
    return SimpleNamespace(url="", institutions=[f"Inst {tier}"], matched=[], tier=tier, source="html")


def _ranks(entries):
    return [e.rank for e in entries]


def test_tier_beats_h_index_and_rank():
    keep = [_entry(1, 90), _entry(2, 80), _entry(3, 5), _entry(4, 1)]
    affs = {keep[2].url: _aff(2), keep[3].url: _aff(1)}
    selected, cut = select_papers(keep, affs, 2)
    assert _ranks(selected) == [3, 4]  # tier 1 (#4) and tier 2 (#3), returned in report order
    assert _ranks(cut) == [1, 2]  # tier 0, best h-index first


def test_tier_order_is_1_then_2_then_0():
    keep = [_entry(1, 50), _entry(2, 50), _entry(3, 50)]
    affs = {keep[0].url: _aff(0), keep[1].url: _aff(2), keep[2].url: _aff(1)}
    ranked = sorted(keep, key=lambda e: ranking_key(e, affs))
    assert _ranks(ranked) == [3, 2, 1]


def test_h_index_then_rank_within_a_tier_none_last():
    keep = [_entry(1, None), _entry(2, 10), _entry(3, 40), _entry(4, 40), _entry(5, 0)]
    selected, cut = select_papers(keep, {}, 3)
    assert _ranks(selected) == [2, 3, 4]  # h 40 (#3), h 40 (#4), h 10 (#2), back in report order
    assert _ranks(cut) == [5, 1]  # ranking order: h 0 before unknown


def test_selected_in_report_order_cut_in_ranking_order():
    keep = [_entry(r, h) for r, h in [(1, 5), (2, 70), (3, 10), (4, 90), (5, 30), (6, 60)]]
    selected, cut = select_papers(keep, {}, 3)
    assert _ranks(selected) == [2, 4, 6]
    assert _ranks(cut) == [5, 3, 1]


@pytest.mark.parametrize("count", [0, 1, 14, 15])
def test_at_or_under_the_cap_is_unchanged(count):
    keep = [_entry(r, 100 - r) for r in range(1, count + 1)]
    selected, cut = select_papers(keep, {}, 15)
    assert selected == keep and cut == []
    assert selected is not keep  # a copy: callers may change it


@pytest.mark.parametrize("max_papers", [0, -1])
def test_zero_or_negative_means_no_cap(max_papers):
    keep = [_entry(r) for r in range(1, 31)]
    selected, cut = select_papers(keep, {keep[0].url: _aff(1)}, max_papers)
    assert selected == keep and cut == []


def test_missing_or_odd_affiliations_count_as_tier_0():
    keep = [_entry(1, 10), _entry(2, 20), _entry(3, 30)]
    affs = {
        keep[0].url: _aff(1),
        keep[1].url: SimpleNamespace(institutions=[]),  # no tier attribute
        # keep[2] missing entirely
    }
    selected, cut = select_papers(keep, affs, 2)
    assert _ranks(selected) == [1, 3]  # tier 1, then the higher h-index of the two tier-0 papers
    assert _ranks(cut) == [2]
    assert tier_of(None) == 0
    assert tier_of(SimpleNamespace(tier=7)) == 0
    assert tier_of(SimpleNamespace(tier="2")) == 2
    assert tier_of(SimpleNamespace(tier=None)) == 0


def test_none_affiliations_mapping_is_allowed():
    keep = [_entry(1, 1), _entry(2, 2)]
    selected, cut = select_papers(keep, None, 1)
    assert _ranks(selected) == [2] and _ranks(cut) == [1]
