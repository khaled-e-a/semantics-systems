"""Small client for Substack's unofficial draft API.

Substack has no official API. We authenticate with the browser's full
``Cookie:`` request header (``SUBSTACK_COOKIE``, copied from DevTools while
logged in); email/password login is captcha-blocked from cloud IPs.

Substack sits behind Cloudflare, which reportedly blocks datacenter IPs such
as GitHub Actions runners. The session therefore uses ``curl_cffi`` with a
Chrome TLS fingerprint and can route through a proxy (``SUBSTACK_PROXY``, e.g.
Cloudflare WARP at ``socks5h://127.0.0.1:40000`` or a residential proxy).
Cloudflare challenge pages and 401/403s raise ``SubstackAuthOrBlockError``
whose message says which of the two it looks like.

Requests are throttled to about 1/s and retried with exponential backoff on
429, 502/503/504 and connection errors. ``create_draft`` is not idempotent, so
it is retried only on 429 and on connection errors raised before the request
reached the server, never on 5xx or read timeouts.

Smoke test (profile → image upload → create draft → delete draft):
    python -m src.substack_post.substack_client --smoke-test
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from curl_cffi.const import CurlECode
from curl_cffi.requests import exceptions as cffi_exc

from .models import DraftResult, UploadedImage

logger = logging.getLogger(__name__)

PROFILE_URL = "https://substack.com/api/v1/user/profile/self"
TIMEOUT_S = 30
UPLOAD_TIMEOUT_S = 60
MAX_IMAGE_BYTES = 1_000_000  # Substack reportedly rejects images of ~1 MB and more
MAX_RETRIES = 3  # retries after the first attempt, sleeping 2, 4, 8 s
BACKOFF_BASE_S = 2.0
MAX_RETRY_AFTER_S = 60.0
MIN_INTERVAL_S = 1.0  # rate limits are unpublished; stay under ~1 request/s
RETRY_STATUSES = {429, 502, 503, 504}
BODY_SNIPPET_CHARS = 300

CLOUDFLARE_MARKERS = (
    "just a moment",
    "cf-chl",
    "challenge-platform",
    "attention required",
    "error code: 1010",
    "error code: 1020",
)
CLOUDFLARE_HINT = (
    "Cloudflare is blocking this IP — set repo variable SUBSTACK_USE_WARP=true or SUBSTACK_PROXY"
)
COOKIE_HINT = (
    "cookie expired or invalid — refresh SUBSTACK_COOKIE (copy the full Cookie request "
    "header from DevTools while logged in to Substack)"
)
_URL_SIZE_RE = re.compile(r"_(\d+)x(\d+)\.\w+$")
_MAGIC_MIME = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)

SMOKE_TEST_TITLE = "[smoke test] delete me"


class SubstackError(Exception):
    """Any failed Substack call. ``status_code`` is set when an HTTP response was received."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SubstackAuthOrBlockError(SubstackError):
    """Substack refused us: Cloudflare blocking (``kind == "cloudflare"``) or a bad cookie (``kind == "cookie"``)."""

    def __init__(self, message: str, kind: str, status_code: Optional[int] = None) -> None:
        super().__init__(message, status_code)
        self.kind = kind


class SubstackClient:
    """Authenticated client bound to one publication (``https://<sub>.substack.com``)."""

    def __init__(
        self,
        publication_url: str,
        cookie: str,
        proxy: Optional[str] = None,
        session: Any = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        min_interval_s: float = MIN_INTERVAL_S,
    ) -> None:
        self.publication_url = _normalize_publication_url(publication_url)
        self._cookie = _normalize_cookie(cookie)
        self._sleep = sleep
        self._clock = clock
        self._min_interval_s = min_interval_s
        self._last_request_at: Optional[float] = None
        self._user_id: Optional[int] = None
        self._headers = {
            "Accept": "application/json",
            "Cookie": self._cookie,
            "Referer": f"{self.publication_url}/publish/home",
            "Origin": self.publication_url,
        }
        if session is None:
            from curl_cffi import requests as cffi_requests

            kwargs: Dict[str, Any] = {"impersonate": "chrome"}
            if proxy:
                kwargs["proxies"] = {"http": proxy, "https": proxy}
            session = cffi_requests.Session(**kwargs)
        self._session = session

    # ---- public API -------------------------------------------------------

    def get_user_id(self) -> int:
        """Return the logged-in user's id (cached after the first call)."""
        if self._user_id is None:
            data = self._request("GET", PROFILE_URL)
            user_id = _as_int(data.get("id")) if isinstance(data, dict) else None
            if user_id is None:
                raise SubstackError("Substack profile response has no numeric 'id'")
            self._user_id = user_id
        return self._user_id

    def upload_image(self, path: Path) -> UploadedImage:
        """Upload a local PNG/JPEG/GIF/WebP (< 1 MB) and return its CDN location and size."""
        path = Path(path)
        raw = path.read_bytes()
        if len(raw) > MAX_IMAGE_BYTES:
            raise SubstackError(
                f"{path.name} is {len(raw):,} bytes; Substack image uploads must be under "
                f"{MAX_IMAGE_BYTES:,} bytes — compress or resize it first"
            )
        mime = _sniff_mime(path, raw)
        data_uri = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

        # Duplicate uploads only leave an orphaned file on the CDN, so 5xx may be retried.
        data = self._request(
            "POST",
            f"{self.publication_url}/api/v1/image",
            payload={"image": data_uri},
            timeout=UPLOAD_TIMEOUT_S,
        )
        if not isinstance(data, dict) or not data.get("url"):
            raise SubstackError("Substack image upload response has no 'url'")
        url = str(data["url"])

        width = _as_int(data.get("imageWidth")) or _as_int(data.get("width"))
        height = _as_int(data.get("imageHeight")) or _as_int(data.get("height"))
        if not (width and height):
            match = _URL_SIZE_RE.search(urlsplit(url).path)
            if match:
                width, height = int(match.group(1)), int(match.group(2))
        if not (width and height):
            width, height = _local_image_size(path)

        return UploadedImage(
            url=url,
            width=width,
            height=height,
            bytes=_as_int(data.get("bytes")) or len(raw),
            content_type=str(data.get("contentType") or data.get("content_type") or mime),
        )

    def create_draft(self, title: str, subtitle: str, doc: dict) -> DraftResult:
        """Create a new newsletter draft authored by the logged-in user."""
        payload = {
            "draft_title": title,
            "draft_subtitle": subtitle,
            "draft_body": json.dumps(doc),  # Substack wants a JSON *string* here
            "draft_bylines": [{"id": self.get_user_id(), "is_guest": False}],
            "audience": "everyone",
            "write_comment_permissions": "everyone",
            "section_chosen": True,
            "draft_section_id": None,
            "type": "newsletter",
        }
        data = self._request(
            "POST", f"{self.publication_url}/api/v1/drafts", payload=payload, idempotent=False
        )
        draft_id = _as_int(data.get("id")) if isinstance(data, dict) else None
        if draft_id is None:
            raise SubstackError("Substack create-draft response has no numeric 'id'")
        logger.info("Created Substack draft %s", draft_id)
        return DraftResult(draft_id=draft_id, edit_url=self.edit_url(draft_id), created=True)

    def update_draft(self, draft_id: int, title: str, subtitle: str, doc: dict) -> DraftResult:
        """Replace the title, subtitle and body of an existing draft."""
        payload = {
            "draft_title": title,
            "draft_subtitle": subtitle,
            "draft_body": json.dumps(doc),
        }
        data = self._request("PUT", f"{self.publication_url}/api/v1/drafts/{draft_id}", payload=payload)
        returned_id = _as_int(data.get("id")) if isinstance(data, dict) else None
        final_id = returned_id if returned_id is not None else int(draft_id)
        logger.info("Updated Substack draft %s", final_id)
        return DraftResult(draft_id=final_id, edit_url=self.edit_url(final_id), created=False)

    def delete_draft(self, draft_id: int) -> None:
        self._request("DELETE", f"{self.publication_url}/api/v1/drafts/{draft_id}", expect_json=False)
        logger.info("Deleted Substack draft %s", draft_id)

    def edit_url(self, draft_id: int) -> str:
        return f"{self.publication_url}/publish/post/{draft_id}"

    # ---- transport --------------------------------------------------------

    def _request(
        self,
        method: str,
        url: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        timeout: float = TIMEOUT_S,
        idempotent: bool = True,
        expect_json: bool = True,
    ) -> Any:
        attempt = 0
        while True:
            self._throttle()
            kwargs: Dict[str, Any] = {
                "headers": dict(self._headers),
                "timeout": timeout,
                # Never follow redirects: a POST would silently turn into a GET, and a
                # redirect to a custom domain ends in Cloudflare 1010 or 401 anyway.
                "allow_redirects": False,
            }
            if payload is not None:
                kwargs["json"] = payload
            try:
                resp = self._session.request(method, url, **kwargs)
            except (cffi_exc.RequestException, OSError) as exc:
                retryable = _is_transient(exc) if idempotent else _is_pre_send(exc)
                if retryable and attempt < MAX_RETRIES:
                    delay = BACKOFF_BASE_S * (2**attempt)
                    logger.warning(
                        "%s %s failed (%s); retry %d/%d in %.0fs",
                        method, url, type(exc).__name__, attempt + 1, MAX_RETRIES, delay,
                    )
                    self._sleep(delay)
                    attempt += 1
                    continue
                raise SubstackError(self._redact(f"{method} {url} failed: {exc}")) from exc

            status = int(resp.status_code)
            text = _response_text(resp)
            logger.debug("%s %s -> %s", method, url, status)

            retryable = status in RETRY_STATUSES and (idempotent or status == 429)
            if retryable and attempt < MAX_RETRIES and not _looks_like_cloudflare(text):
                delay = BACKOFF_BASE_S * (2**attempt)
                if status == 429:
                    delay = max(delay, _retry_after_s(resp))
                logger.warning(
                    "%s %s returned HTTP %d; retry %d/%d in %.0fs",
                    method, url, status, attempt + 1, MAX_RETRIES, delay,
                )
                self._sleep(delay)
                attempt += 1
                continue

            return self._handle_response(method, url, status, text, resp, expect_json)

    def _handle_response(
        self, method: str, url: str, status: int, text: str, resp: Any, expect_json: bool
    ) -> Any:
        where = f"{method} {url}"
        if status in (403, 503) and _looks_like_cloudflare(text):
            raise SubstackAuthOrBlockError(
                f"Substack blocked {where} (HTTP {status}, Cloudflare challenge/block page): "
                f"{CLOUDFLARE_HINT}",
                kind="cloudflare",
                status_code=status,
            )
        if status in (401, 403):
            raise SubstackAuthOrBlockError(
                f"Substack rejected {where} (HTTP {status}): {COOKIE_HINT}",
                kind="cookie",
                status_code=status,
            )
        if 300 <= status < 400:
            location = _header(resp, "location") or "?"
            if re.search(r"sign-?in|login", location, re.IGNORECASE):
                raise SubstackAuthOrBlockError(
                    f"Substack redirected {where} to a login page: {COOKIE_HINT}",
                    kind="cookie",
                    status_code=status,
                )
            raise SubstackError(
                self._redact(
                    f"Substack redirected {where} (HTTP {status}) to {location} — "
                    "SUBSTACK_PUBLICATION_URL must be the canonical https://<sub>.substack.com"
                ),
                status_code=status,
            )
        if not 200 <= status < 300:
            raise SubstackError(
                self._redact(f"Substack {where} failed with HTTP {status}: {_snippet(text)}"),
                status_code=status,
            )

        body = text.strip()
        if not body:
            if expect_json:
                raise SubstackError(f"Substack {where} returned an empty body", status_code=status)
            return None
        try:
            return json.loads(body)
        except ValueError:
            pass
        if _looks_like_cloudflare(body):
            raise SubstackAuthOrBlockError(
                f"Substack {where} returned a Cloudflare challenge page (HTTP {status}): "
                f"{CLOUDFLARE_HINT}",
                kind="cloudflare",
                status_code=status,
            )
        if body.startswith("<"):
            raise SubstackAuthOrBlockError(
                f"Substack {where} returned an HTML page instead of JSON (HTTP {status}), "
                f"most likely a login page: {COOKIE_HINT}. If the cookie is fresh, "
                "check that SUBSTACK_PUBLICATION_URL is https://<sub>.substack.com",
                kind="cookie",
                status_code=status,
            )
        if expect_json:
            raise SubstackError(
                self._redact(f"Substack {where} returned non-JSON: {_snippet(body)}"),
                status_code=status,
            )
        return None

    def _throttle(self) -> None:
        if self._last_request_at is not None and self._min_interval_s > 0:
            wait = self._last_request_at + self._min_interval_s - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._clock()

    def _redact(self, message: str) -> str:
        return redact_cookie(message, self._cookie)


# ---- helpers ---------------------------------------------------------------


def redact_cookie(message: str, cookie: str) -> str:
    """Remove the cookie header and every individual cookie value from ``message``."""
    if not cookie:
        return message
    secrets: List[str] = [cookie]
    for part in cookie.split(";"):
        _, _, value = part.partition("=")
        value = value.strip()
        if len(value) >= 6:
            secrets.append(value)
    for secret in sorted(set(secrets), key=len, reverse=True):
        message = message.replace(secret, "[redacted]")
    return message


def _normalize_publication_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise ValueError("SUBSTACK_PUBLICATION_URL is empty")
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    parts = urlsplit(url)
    if not parts.netloc:
        raise ValueError(f"SUBSTACK_PUBLICATION_URL is not a URL: {url!r}")
    host = parts.netloc.lower()
    if not host.endswith(".substack.com"):
        logger.warning(
            "SUBSTACK_PUBLICATION_URL host %s is not *.substack.com; custom domains can be "
            "blocked by Cloudflare (error 1010). Use https://<sub>.substack.com.",
            host,
        )
    return f"https://{host}"


def _normalize_cookie(cookie: str) -> str:
    cookie = " ".join((cookie or "").replace("\r", " ").replace("\n", " ").split())
    if cookie.lower().startswith("cookie:"):
        cookie = cookie[len("cookie:") :].strip()
    if not cookie:
        raise ValueError("SUBSTACK_COOKIE is empty")
    if "substack.sid" not in cookie and "connect.sid" not in cookie:
        logger.warning(
            "SUBSTACK_COOKIE has neither substack.sid nor connect.sid; copy the full Cookie "
            "request header from DevTools while logged in."
        )
    return cookie


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, (cffi_exc.Timeout, cffi_exc.ConnectionError, cffi_exc.ProxyError))


def _is_pre_send(exc: BaseException) -> bool:
    """True when the request certainly never reached Substack (safe to retry a POST)."""
    if isinstance(exc, (cffi_exc.DNSError, cffi_exc.SSLError, cffi_exc.ProxyError)):
        return True
    return getattr(exc, "code", None) in (CurlECode.COULDNT_CONNECT, CurlECode.QUIC_CONNECT_ERROR)


def _looks_like_cloudflare(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in CLOUDFLARE_MARKERS)


def _response_text(resp: Any) -> str:
    try:
        text = resp.text
    except Exception:  # undecodable body
        return ""
    return text if isinstance(text, str) else ""


def _header(resp: Any, name: str) -> Optional[str]:
    headers = getattr(resp, "headers", None) or {}
    try:
        value = headers.get(name)
        if value is None:
            value = headers.get(name.title())
    except Exception:
        return None
    return str(value) if value is not None else None


def _retry_after_s(resp: Any) -> float:
    value = _header(resp, "retry-after")
    try:
        return min(float(value), MAX_RETRY_AFTER_S) if value else 0.0
    except ValueError:
        return 0.0


def _snippet(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) > BODY_SNIPPET_CHARS:
        return collapsed[:BODY_SNIPPET_CHARS] + "…"
    return collapsed


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sniff_mime(path: Path, raw: bytes) -> str:
    for magic, mime in _MAGIC_MIME:
        if raw.startswith(magic):
            return mime
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed and guessed.startswith("image/"):
        return guessed
    raise SubstackError(f"{path.name} is not a PNG, JPEG, GIF or WebP image")


def _local_image_size(path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception as exc:
        raise SubstackError(f"could not determine the size of {path.name}: {exc}") from exc


# ---- smoke test CLI --------------------------------------------------------


def _load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _write_tiny_png(path: Path) -> None:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (160, 90), (37, 99, 235))
    ImageDraw.Draw(img).rectangle((20, 20, 140, 70), outline=(255, 255, 255), width=4)
    img.save(path, "PNG")


def run_smoke_test(client: SubstackClient) -> str:
    """Profile → upload → create draft → delete draft. Returns a one-line summary."""
    from .prosemirror import markdown_to_doc

    user_id = client.get_user_id()
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "smoke-test.png"
        _write_tiny_png(png)
        image = client.upload_image(png)

    body = (
        "This draft was created by the **vv-paper-fetcher** Substack smoke test. "
        "It should have been deleted automatically; delete it if you can see it.\n\n"
        f"![Smoke test image]({png.name})"
    )
    doc = markdown_to_doc(body, {png.name: image})
    draft = client.create_draft(SMOKE_TEST_TITLE, "Automated connectivity check", doc)
    try:
        client.delete_draft(draft.draft_id)
    except SubstackError:
        print(f"Could not delete the smoke-test draft; delete it by hand: {draft.edit_url}", file=sys.stderr)
        raise

    return (
        f"Substack smoke test passed for {client.publication_url}: user id {user_id}; "
        f"image uploaded ({image.width}x{image.height}, {image.bytes} bytes, {image.url}); "
        f"draft {draft.draft_id} created and deleted."
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Substack draft API client.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="fetch the profile, upload a tiny PNG, create a draft and delete it",
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.smoke_test:
        parser.print_help(sys.stderr)
        return 1

    _load_env()
    publication_url = os.environ.get("SUBSTACK_PUBLICATION_URL", "")
    cookie = os.environ.get("SUBSTACK_COOKIE", "")
    proxy = os.environ.get("SUBSTACK_PROXY") or None
    missing = [
        name
        for name, value in (("SUBSTACK_PUBLICATION_URL", publication_url), ("SUBSTACK_COOKIE", cookie))
        if not value.strip()
    ]
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}", file=sys.stderr)
        return 1

    try:
        client = SubstackClient(publication_url, cookie, proxy=proxy)
        summary = run_smoke_test(client)
    except SubstackAuthOrBlockError as exc:
        print(f"Substack smoke test FAILED: {redact_cookie(str(exc), cookie)}", file=sys.stderr)
        hint = CLOUDFLARE_HINT if exc.kind == "cloudflare" else "refresh SUBSTACK_COOKIE"
        print(f"Hint: {hint}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            f"Substack smoke test FAILED: {type(exc).__name__}: {redact_cookie(str(exc), cookie)}",
            file=sys.stderr,
        )
        return 1

    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
