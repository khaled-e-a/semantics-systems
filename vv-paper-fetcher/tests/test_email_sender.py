import base64
from unittest.mock import MagicMock, patch

from src.email_sender import attachment_from_path, send_digest_email


def _ok():
    resp = MagicMock()
    resp.status_code = 200
    return resp


@patch("src.email_sender.requests.post")
def test_payload_without_attachments_is_unchanged(mock_post):
    mock_post.return_value = _ok()
    assert send_digest_email("key", "from@x.com", "a@x.com, b@x.com", "Subj", "<p>hi</p>") is True
    payload = mock_post.call_args.kwargs["json"]
    assert payload == {"from": "from@x.com", "to": ["a@x.com", "b@x.com"], "subject": "Subj", "html": "<p>hi</p>"}
    assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer key"


@patch("src.email_sender.requests.post")
def test_attachments_are_passed_to_resend(mock_post, tmp_path):
    mock_post.return_value = _ok()
    fig = tmp_path / "figure.png"
    fig.write_bytes(b"\x89PNG\r\n\x1a\nDATA")

    attachment = attachment_from_path(fig, "fig-2609.00001.png")
    assert attachment["filename"] == "fig-2609.00001.png"
    assert base64.b64decode(attachment["content"]) == b"\x89PNG\r\n\x1a\nDATA"
    assert attachment_from_path(fig)["filename"] == "figure.png"

    assert send_digest_email("key", "f@x.com", "t@x.com", "S", "<p/>", attachments=[attachment]) is True
    assert mock_post.call_args.kwargs["json"]["attachments"] == [attachment]


@patch("src.email_sender.requests.post")
def test_send_failure_returns_false(mock_post):
    resp = MagicMock()
    resp.status_code = 422
    resp.text = "bad attachment"
    mock_post.return_value = resp
    assert send_digest_email("key", "f@x.com", "t@x.com", "S", "<p/>", attachments=[]) is False
