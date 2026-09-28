import base64
import json
from pathlib import Path

import pytest
from curl_cffi.const import CurlECode
from curl_cffi.requests import exceptions as cffi_exc
from PIL import Image

from src.substack_post import substack_client as sc
from src.substack_post.models import DraftResult, UploadedImage
from src.substack_post.substack_client import (
    PROFILE_URL,
    SubstackAuthOrBlockError,
    SubstackClient,
    SubstackError,
)

PUB = "https://vvuq.substack.com"
SID = "s%3AsuperSecretSessionValue123"
COOKIE = f"substack.sid={SID}; ajs_anonymous_id=abcdef123456"
DOC = {"type": "doc", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Hi"}]}]}
CLOUDFLARE_PAGE = (
    "<!DOCTYPE html><html><head><title>Just a moment...</title></head>"
    "<body><script src='/cdn-cgi/challenge-platform/h/b/orchestrate/chl_page'></script></body></html>"
)


class FakeResponse:
    def __init__(self, status=200, body=None, text=None, headers=None):
        self.status_code = status
        if text is None:
            text = json.dumps(body) if body is not None else ""
        self.text = text
        self.headers = headers or {}


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def make_client(*responses, pub=PUB):
    session = FakeSession(*responses)
    sleeps = []
    client = SubstackClient(pub, COOKIE, session=session, sleep=sleeps.append, min_interval_s=0)
    return client, session, sleeps


def write_png(path: Path, size=(40, 30)) -> Path:
    Image.new("RGB", size, (255, 0, 0)).save(path, "PNG")
    return path


# ---- construction ----------------------------------------------------------


def test_publication_url_is_normalized():
    client, _, _ = make_client(pub="vvuq.substack.com/publish/home/")
    assert client.publication_url == PUB
    client, _, _ = make_client(pub="https://VVUQ.substack.com/")
    assert client.publication_url == PUB


def test_empty_cookie_rejected():
    with pytest.raises(ValueError):
        SubstackClient(PUB, "  ", session=FakeSession())


def test_real_session_uses_chrome_impersonation_and_proxy(monkeypatch):
    import curl_cffi.requests

    created = {}

    class RecordingSession:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(curl_cffi.requests, "Session", RecordingSession)
    SubstackClient(PUB, COOKIE, proxy="socks5h://127.0.0.1:40000")
    assert created["impersonate"] == "chrome"
    assert created["proxies"] == {"http": "socks5h://127.0.0.1:40000", "https": "socks5h://127.0.0.1:40000"}

    created.clear()
    SubstackClient(PUB, COOKIE)
    assert "proxies" not in created


# ---- profile ---------------------------------------------------------------


def test_get_user_id_sends_browser_headers_and_caches():
    client, session, _ = make_client(FakeResponse(body={"id": 4242, "name": "K"}))
    assert client.get_user_id() == 4242
    assert client.get_user_id() == 4242
    assert len(session.calls) == 1

    method, url, kwargs = session.calls[0]
    assert (method, url) == ("GET", PROFILE_URL)
    assert kwargs["headers"] == {
        "Accept": "application/json",
        "Cookie": COOKIE,
        "Referer": f"{PUB}/publish/home",
        "Origin": PUB,
    }
    assert kwargs["timeout"] == 30
    assert kwargs["allow_redirects"] is False


def test_profile_without_id_is_an_error():
    client, _, _ = make_client(FakeResponse(body={"name": "no id"}))
    with pytest.raises(SubstackError):
        client.get_user_id()


# ---- image upload ----------------------------------------------------------


def test_upload_builds_data_uri_and_reads_image_width_fields(tmp_path):
    png = write_png(tmp_path / "fig.png")
    resp = {
        "url": "https://substack-post-media.s3.amazonaws.com/public/images/u_40x30.png",
        "imageWidth": 1200,
        "imageHeight": 800,
        "bytes": 999,
        "contentType": "image/png",
    }
    client, session, _ = make_client(FakeResponse(body=resp))
    image = client.upload_image(png)

    assert image == UploadedImage(url=resp["url"], width=1200, height=800, bytes=999, content_type="image/png")
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("POST", f"{PUB}/api/v1/image")
    assert kwargs["timeout"] == 60
    data_uri = kwargs["json"]["image"]
    prefix = "data:image/png;base64,"
    assert data_uri.startswith(prefix)
    assert base64.b64decode(data_uri[len(prefix) :]) == png.read_bytes()


def test_upload_reads_plain_width_height_fields(tmp_path):
    png = write_png(tmp_path / "fig.png")
    client, _, _ = make_client(FakeResponse(body={"url": "https://cdn/x.png", "width": 640, "height": 480}))
    image = client.upload_image(png)
    assert (image.width, image.height) == (640, 480)
    assert image.bytes == png.stat().st_size  # falls back to the local size
    assert image.content_type == "image/png"


def test_upload_falls_back_to_url_suffix(tmp_path):
    png = write_png(tmp_path / "fig.png")
    url = "https://substack-post-media.s3.amazonaws.com/public/images/abc-def_1456x819.png"
    client, _, _ = make_client(FakeResponse(body={"url": url, "bytes": 5}))
    image = client.upload_image(png)
    assert (image.width, image.height) == (1456, 819)


def test_upload_falls_back_to_local_file_size(tmp_path):
    png = write_png(tmp_path / "fig.png", size=(33, 22))
    client, _, _ = make_client(FakeResponse(body={"url": "https://cdn/no-size.png"}))
    image = client.upload_image(png)
    assert (image.width, image.height) == (33, 22)


def test_upload_detects_jpeg(tmp_path):
    jpg = tmp_path / "fig.jpg"
    Image.new("RGB", (10, 10)).save(jpg, "JPEG")
    client, session, _ = make_client(FakeResponse(body={"url": "https://cdn/x_10x10.jpg"}))
    image = client.upload_image(jpg)
    assert session.calls[0][2]["json"]["image"].startswith("data:image/jpeg;base64,")
    assert image.content_type == "image/jpeg"


def test_upload_rejects_files_over_1mb_before_any_request(tmp_path):
    big = tmp_path / "big.png"
    big.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 1_000_000)
    client, session, _ = make_client()
    with pytest.raises(SubstackError, match="1,000,000"):
        client.upload_image(big)
    assert session.calls == []


# ---- drafts ----------------------------------------------------------------


def test_create_draft_payload_and_result():
    client, session, _ = make_client(FakeResponse(body={"id": 7}), FakeResponse(body={"id": 555}))
    result = client.create_draft("Title", "Sub", DOC)

    assert result == DraftResult(draft_id=555, edit_url=f"{PUB}/publish/post/555", created=True)
    method, url, kwargs = session.calls[1]
    assert (method, url) == ("POST", f"{PUB}/api/v1/drafts")
    payload = kwargs["json"]
    assert isinstance(payload["draft_body"], str)
    assert json.loads(payload["draft_body"]) == DOC
    assert payload["draft_bylines"] == [{"id": 7, "is_guest": False}]
    assert payload["draft_title"] == "Title" and payload["draft_subtitle"] == "Sub"
    assert payload["type"] == "newsletter"
    assert payload["audience"] == "everyone"
    assert payload["write_comment_permissions"] == "everyone"
    assert payload["section_chosen"] is True and payload["draft_section_id"] is None


def test_update_draft_uses_put_to_the_id():
    client, session, _ = make_client(FakeResponse(body={"id": 555, "draft_title": "New"}))
    result = client.update_draft(555, "New", "Sub", DOC)

    assert result == DraftResult(draft_id=555, edit_url=f"{PUB}/publish/post/555", created=False)
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("PUT", f"{PUB}/api/v1/drafts/555")
    assert isinstance(kwargs["json"]["draft_body"], str)
    assert json.loads(kwargs["json"]["draft_body"]) == DOC
    assert kwargs["json"]["draft_title"] == "New"


def test_update_missing_draft_exposes_status_code():
    client, _, _ = make_client(FakeResponse(404, body={"error": "Not found"}))
    with pytest.raises(SubstackError) as info:
        client.update_draft(1, "t", "s", DOC)
    assert info.value.status_code == 404
    assert not isinstance(info.value, SubstackAuthOrBlockError)


def test_delete_draft_accepts_empty_body():
    client, session, _ = make_client(FakeResponse(200, text=""))
    assert client.delete_draft(555) is None
    assert session.calls[0][:2] == ("DELETE", f"{PUB}/api/v1/drafts/555")


def test_edit_url():
    client, _, _ = make_client(pub=PUB + "/")
    assert client.edit_url(123) == "https://vvuq.substack.com/publish/post/123"


# ---- auth / Cloudflare detection -------------------------------------------


def test_cloudflare_html_403_is_a_block_error():
    client, session, sleeps = make_client(FakeResponse(403, text=CLOUDFLARE_PAGE, headers={"content-type": "text/html"}))
    with pytest.raises(SubstackAuthOrBlockError) as info:
        client.get_user_id()
    assert info.value.kind == "cloudflare"
    assert "Cloudflare is blocking this IP" in str(info.value)
    assert "SUBSTACK_USE_WARP=true" in str(info.value)
    assert len(session.calls) == 1 and sleeps == []


def test_cloudflare_1010_and_503_challenge_are_not_retried():
    client, session, _ = make_client(FakeResponse(403, text="error code: 1010"))
    with pytest.raises(SubstackAuthOrBlockError, match="Cloudflare"):
        client.get_user_id()

    client, session, sleeps = make_client(FakeResponse(503, text=CLOUDFLARE_PAGE))
    with pytest.raises(SubstackAuthOrBlockError) as info:
        client.get_user_id()
    assert info.value.kind == "cloudflare"
    assert len(session.calls) == 1 and sleeps == []


def test_401_is_a_cookie_error():
    client, _, _ = make_client(FakeResponse(401, body={"error": "Not authorized"}))
    with pytest.raises(SubstackAuthOrBlockError) as info:
        client.get_user_id()
    assert info.value.kind == "cookie"
    assert info.value.status_code == 401
    assert "refresh SUBSTACK_COOKIE" in str(info.value)
    assert "Cloudflare" not in str(info.value)


def test_plain_403_is_a_cookie_error():
    client, _, _ = make_client(FakeResponse(403, body={"error": "Forbidden"}))
    with pytest.raises(SubstackAuthOrBlockError) as info:
        client.create_draft("t", "s", DOC)
    assert info.value.kind == "cookie"


def test_html_200_on_api_endpoint_is_an_auth_or_block_error():
    client, _, _ = make_client(FakeResponse(200, text="<!doctype html><html><body>Sign in</body></html>"))
    with pytest.raises(SubstackAuthOrBlockError) as info:
        client.get_user_id()
    assert info.value.kind == "cookie"

    client, _, _ = make_client(FakeResponse(200, text=CLOUDFLARE_PAGE))
    with pytest.raises(SubstackAuthOrBlockError) as info:
        client.get_user_id()
    assert info.value.kind == "cloudflare"


def test_redirect_is_not_followed_and_explains_canonical_url():
    client, session, _ = make_client(
        FakeResponse(301, text="", headers={"location": "https://custom.example.com/api/v1/drafts/1"})
    )
    with pytest.raises(SubstackError, match="canonical") as info:
        client.update_draft(1, "t", "s", DOC)
    assert info.value.status_code == 301
    assert session.calls[0][2]["allow_redirects"] is False


def test_redirect_to_sign_in_is_a_cookie_error():
    client, _, _ = make_client(FakeResponse(302, text="", headers={"location": "https://substack.com/sign-in?redirect=x"}))
    with pytest.raises(SubstackAuthOrBlockError) as info:
        client.get_user_id()
    assert info.value.kind == "cookie"


# ---- retries ---------------------------------------------------------------


def test_429_then_200_is_retried_with_backoff():
    client, session, sleeps = make_client(FakeResponse(429, text="slow down"), FakeResponse(body={"id": 9}))
    assert client.get_user_id() == 9
    assert len(session.calls) == 2
    assert sleeps == [2.0]


def test_429_honours_retry_after():
    client, _, sleeps = make_client(
        FakeResponse(429, text="", headers={"retry-after": "7"}), FakeResponse(body={"id": 9})
    )
    client.get_user_id()
    assert sleeps == [7.0]


def test_create_draft_retried_on_429():
    client, session, _ = make_client(
        FakeResponse(body={"id": 7}), FakeResponse(429, text=""), FakeResponse(body={"id": 555})
    )
    assert client.create_draft("t", "s", DOC).draft_id == 555
    assert [c[0] for c in session.calls] == ["GET", "POST", "POST"]


def test_500_on_create_is_not_retried():
    client, session, sleeps = make_client(FakeResponse(body={"id": 7}), FakeResponse(500, text="Internal error"))
    with pytest.raises(SubstackError) as info:
        client.create_draft("t", "s", DOC)
    assert info.value.status_code == 500
    assert len(session.calls) == 2 and sleeps == []


def test_503_on_create_is_not_retried():
    client, session, _ = make_client(FakeResponse(body={"id": 7}), FakeResponse(503, text="unavailable"))
    with pytest.raises(SubstackError):
        client.create_draft("t", "s", DOC)
    assert len(session.calls) == 2


def test_create_retried_on_connect_error_but_not_on_timeout():
    refused = cffi_exc.ConnectionError("Failed to connect", code=CurlECode.COULDNT_CONNECT)
    client, session, sleeps = make_client(FakeResponse(body={"id": 7}), refused, FakeResponse(body={"id": 555}))
    assert client.create_draft("t", "s", DOC).draft_id == 555
    assert sleeps == [2.0]

    timeout = cffi_exc.Timeout("Operation timed out", code=CurlECode.OPERATION_TIMEDOUT)
    client, session, sleeps = make_client(FakeResponse(body={"id": 7}), timeout)
    with pytest.raises(SubstackError, match="timed out"):
        client.create_draft("t", "s", DOC)
    assert len(session.calls) == 2 and sleeps == []


def test_idempotent_calls_retry_5xx_and_timeouts_then_give_up():
    timeout = cffi_exc.Timeout("Operation timed out", code=CurlECode.OPERATION_TIMEDOUT)
    client, session, sleeps = make_client(
        FakeResponse(502, text="bad gateway"),
        timeout,
        FakeResponse(504, text="gateway timeout"),
        FakeResponse(503, text="unavailable"),
    )
    with pytest.raises(SubstackError) as info:
        client.get_user_id()
    assert info.value.status_code == 503
    assert len(session.calls) == 4
    assert sleeps == [2.0, 4.0, 8.0]


def test_requests_are_throttled_to_one_per_second():
    now = [100.0]
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    session = FakeSession(FakeResponse(body={"id": 1}), FakeResponse(text=""))
    client = SubstackClient(PUB, COOKIE, session=session, sleep=fake_sleep, clock=lambda: now[0])
    client.get_user_id()
    now[0] += 0.25
    client.delete_draft(3)
    assert sleeps == [pytest.approx(0.75)]


# ---- secrecy ---------------------------------------------------------------


def test_cookie_never_appears_in_exception_messages():
    echoed = f"<pre>debug: Cookie: {COOKIE}</pre>"
    client, _, _ = make_client(FakeResponse(500, text=echoed))
    with pytest.raises(SubstackError) as info:
        client.update_draft(1, "t", "s", DOC)
    assert SID not in str(info.value) and COOKIE not in str(info.value)

    client, _, _ = make_client(OSError(f"proxy said {SID}"))
    with pytest.raises(SubstackError) as info:
        client.delete_draft(1)
    assert SID not in str(info.value)

    for status, text in ((401, ""), (403, CLOUDFLARE_PAGE), (200, "<html>login</html>")):
        client, _, _ = make_client(FakeResponse(status, text=text))
        with pytest.raises(SubstackAuthOrBlockError) as info:
            client.get_user_id()
        assert SID not in str(info.value)


# ---- smoke test ------------------------------------------------------------


def test_run_smoke_test_end_to_end_with_fake_session():
    client, session, _ = make_client(
        FakeResponse(body={"id": 7}),
        FakeResponse(body={"url": "https://cdn/x_160x90.png", "bytes": 321, "contentType": "image/png"}),
        FakeResponse(body={"id": 99}),
        FakeResponse(text=""),
    )
    summary = sc.run_smoke_test(client)

    assert [(m, u) for m, u, _ in session.calls] == [
        ("GET", PROFILE_URL),
        ("POST", f"{PUB}/api/v1/image"),
        ("POST", f"{PUB}/api/v1/drafts"),
        ("DELETE", f"{PUB}/api/v1/drafts/99"),
    ]
    payload = session.calls[2][2]["json"]
    assert payload["draft_title"] == "[smoke test] delete me"
    doc = json.loads(payload["draft_body"])
    assert [n["type"] for n in doc["content"]] == ["paragraph", "captionedImage"]
    assert doc["content"][1]["content"][0]["attrs"]["src"] == "https://cdn/x_160x90.png"
    assert "99" in summary and COOKIE not in summary


class _FakeClient:
    error = None

    def __init__(self, publication_url, cookie, proxy=None, session=None):
        self.publication_url = publication_url
        self.proxy = proxy

    def get_user_id(self):
        if self.error:
            raise self.error
        return 1

    def upload_image(self, path):
        assert Path(path).stat().st_size > 0
        return UploadedImage(url="https://cdn/i_160x90.png", width=160, height=90, bytes=10, content_type="image/png")

    def create_draft(self, title, subtitle, doc):
        return DraftResult(draft_id=2, edit_url=f"{self.publication_url}/publish/post/2", created=True)

    def delete_draft(self, draft_id):
        return None


@pytest.fixture
def smoke_env(monkeypatch):
    monkeypatch.setattr(sc, "_load_env", lambda: None)
    monkeypatch.setattr(sc, "SubstackClient", _FakeClient)
    monkeypatch.setattr(_FakeClient, "error", None)
    monkeypatch.setenv("SUBSTACK_PUBLICATION_URL", PUB)
    monkeypatch.setenv("SUBSTACK_COOKIE", COOKIE)
    monkeypatch.delenv("SUBSTACK_PROXY", raising=False)
    return monkeypatch


def test_smoke_cli_success_exit_0(smoke_env, capsys):
    assert sc.main(["--smoke-test"]) == 0
    out, err = capsys.readouterr()
    assert "smoke test passed" in out
    assert SID not in out + err


def test_smoke_cli_cloudflare_block_exit_2(smoke_env, capsys):
    _FakeClient.error = SubstackAuthOrBlockError("blocked by Cloudflare", kind="cloudflare", status_code=403)
    assert sc.main(["--smoke-test"]) == 2
    err = capsys.readouterr().err
    assert "Cloudflare is blocking this IP — set repo variable SUBSTACK_USE_WARP=true or SUBSTACK_PROXY" in err


def test_smoke_cli_expired_cookie_exit_2(smoke_env, capsys):
    _FakeClient.error = SubstackAuthOrBlockError(f"bad cookie {COOKIE}", kind="cookie", status_code=401)
    assert sc.main(["--smoke-test"]) == 2
    err = capsys.readouterr().err
    assert "refresh SUBSTACK_COOKIE" in err
    assert SID not in err


def test_smoke_cli_other_error_exit_1(smoke_env, capsys):
    _FakeClient.error = SubstackError("HTTP 500")
    assert sc.main(["--smoke-test"]) == 1
    _FakeClient.error = RuntimeError("boom")
    assert sc.main(["--smoke-test"]) == 1


def test_smoke_cli_missing_env_exit_1(smoke_env, capsys):
    smoke_env.delenv("SUBSTACK_COOKIE")
    assert sc.main(["--smoke-test"]) == 1
    assert "SUBSTACK_COOKIE" in capsys.readouterr().err


def test_cli_without_flag_exit_1(smoke_env):
    assert sc.main([]) == 1
