import json
from datetime import datetime, timedelta, timezone

from src.substack_post import draft_state
from src.substack_post.models import DraftResult


def test_missing_file_loads_empty(tmp_path):
    assert draft_state.load_drafts(tmp_path / "nope.json") == {}


def test_record_save_load_roundtrip(tmp_path):
    path = tmp_path / "state" / "substack_drafts.json"
    drafts = draft_state.load_drafts(path)
    now = datetime(2026, 9, 28, 14, 30, tzinfo=timezone(timedelta(hours=-4)))
    draft_state.record_draft(
        drafts, "2026-09-28", DraftResult(123, "https://x.substack.com/publish/post/123", True), "5 Ways “to” Verify", now=now
    )
    draft_state.save_drafts(drafts, path)

    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n") and "“to”" in raw
    loaded = draft_state.load_drafts(path)
    assert loaded == {
        "2026-09-28": {
            "draft_id": 123,
            "edit_url": "https://x.substack.com/publish/post/123",
            "title": "5 Ways “to” Verify",
            "updated_at": "2026-09-28T18:30:00+00:00",
        }
    }
    assert json.loads(raw) == loaded


def test_record_overwrites_same_date_and_keeps_others(tmp_path):
    drafts = {"2026-09-21": {"draft_id": 1, "edit_url": "u1", "title": "a", "updated_at": "t"}}
    draft_state.record_draft(drafts, "2026-09-28", DraftResult(2, "u2", True), "b")
    draft_state.record_draft(drafts, "2026-09-28", DraftResult(2, "u2", False), "c")
    assert set(drafts) == {"2026-09-21", "2026-09-28"}
    assert drafts["2026-09-28"]["title"] == "c"


def test_get_draft():
    drafts = {"2026-09-28": {"draft_id": 7, "edit_url": "u"}, "2026-09-21": {"edit_url": "u"}}
    assert draft_state.get_draft(drafts, "2026-09-28")["draft_id"] == 7
    assert draft_state.get_draft(drafts, "2026-09-21") is None
    assert draft_state.get_draft(drafts, "2026-09-14") is None
