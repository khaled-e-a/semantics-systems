"""Publish a run's figure files to a public ``figures`` branch and return raw URLs.

Substack's editor keeps pasted images only when they are ``<img src="https://...">``
with public URLs, so the emailed post can't carry its figures as attachments.
``publish_figures`` commits them to a dedicated orphan branch of this (public)
repository, one folder per report date, and returns
``https://raw.githubusercontent.com/<owner>/<repo>/<branch>/<date>/<file>`` URLs.

It is written for GitHub Actions after ``actions/checkout@v4``: a shallow
clone with only ``main`` fetched and the push token stored in the checkout's
local git config (``http.https://github.com/.extraheader``). A fresh
``git clone`` would not have that token, so everything happens inside the
existing repository with git plumbing:

1. ``git fetch [--depth=1] <remote> +refs/heads/<branch>:<temp ref>``
   (``--depth=1`` only when the checkout is already shallow, so a full local
   clone is never made shallow). If the fetch fails and ``git ls-remote`` shows
   no such branch, the branch is created as an orphan with a README.md.
2. ``git hash-object -w`` the images, then build the new tree in a TEMPORARY
   index (``GIT_INDEX_FILE``): ``read-tree`` the branch tip, ``update-index``
   the ``<date>/<file>`` entries, ``write-tree``. The caller's working tree,
   index and current branch are never touched.
3. Same tree as the tip: nothing to commit. Otherwise ``commit-tree`` (message
   ``figures: <date>``; the github-actions[bot] identity when user.name/email
   aren't configured) and ``git push <remote> <sha>:refs/heads/<branch>``
   without force. A rejected push (someone else pushed) re-fetches and retries
   once.
4. Poll every URL with ``requests.head`` until it answers 200 with an
   ``image/*`` Content-Type. raw.githubusercontent.com's CDN caches 404s for
   about 5 minutes, so each poll adds a unique query string: the check always
   reaches GitHub and never seeds the bare URL's cache with a 404.

Only plumbing that has existed for years is used; the newest requirement is
``git rev-parse --is-shallow-repository`` (git >= 2.15).

Re-publishing a date with changed bytes overwrites the files with a new
commit, but the CDN may keep serving the old image under the same URL for up
to 5 minutes.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)

RAW_URL_TEMPLATE = "https://raw.githubusercontent.com/{slug}/{branch}/{date}/{filename}"
BOT_NAME = "github-actions[bot]"
BOT_EMAIL = "github-actions[bot]@users.noreply.github.com"
README_TEXT = (
    "# Figures\n"
    "\n"
    "Figures for the weekly Substack posts, published by write_post.py; one folder per report date.\n"
    "\n"
    "Written by `vv-paper-fetcher/src/substack_post/figure_host.py`. Each file is served from\n"
    "`https://raw.githubusercontent.com/<owner>/<repo>/figures/<YYYY-MM-DD>/<file>`, and the\n"
    "posts link to those URLs, so don't move, rename or delete files once a post is out.\n"
)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp")
PUSH_ATTEMPTS = 2  # the first try plus one retry after re-fetching
LOCAL_GIT_TIMEOUT_S = 60
NETWORK_GIT_TIMEOUT_S = 300
HEAD_TIMEOUT_S = 10
POLL_INITIAL_S = 1.0
POLL_MAX_INTERVAL_S = 15.0
STDERR_SNIPPET_CHARS = 600
TEMP_REF_PREFIX = "refs/figure-host/"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SCP_RE = re.compile(r"^(?:[^@/]+@)?(?P<host>[^:/]+):(?P<path>.+)$")
_GITHUB_HOSTS = ("github.com", "www.github.com", "ssh.github.com")
# Whatever git prints is redacted before it reaches an exception message or log line.
_REDACTIONS = (
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s@]+@"), r"\1***@"),
    (re.compile(r"(?i)(authorization:\s*)(?:basic|bearer|token)?\s*\S+"), r"\1***"),
    (re.compile(r"(?i)(extraheader\s*=?\s*)\S.*"), r"\1***"),
    (re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"), "***"),
)


class FigureHostError(Exception):
    """Publishing failed: bad input, a git or push failure, or URLs that never became reachable."""


def publish_figures(
    files: Mapping[str, Tuple[Path, str]],
    report_date: str,
    repo_dir: Path,
    *,
    branch: str = "figures",
    remote: str = "origin",
    repo_slug: Optional[str] = None,
    wait_timeout_s: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Dict[str, str]:
    """Commit ``files`` to ``<branch>/<report_date>/`` on ``remote`` and return ``{paper_url: public URL}``.

    ``files`` maps each paper URL to ``(local image path, destination file name)``.
    ``repo_dir`` is any directory inside the git checkout. ``repo_slug``
    ("owner/name") defaults to ``$GITHUB_REPOSITORY``, else it is parsed from
    ``git remote get-url <remote>``. After pushing, waits up to
    ``wait_timeout_s`` for every URL to serve the image (0 or less: don't
    wait); ``sleep``/``clock`` are injectable for tests.

    Raises ``FigureHostError`` on any failure. Never modifies the caller's
    working tree, index or current branch.
    """
    if not files:
        return {}
    if not _DATE_RE.match(report_date or ""):
        raise FigureHostError(f"report_date must be YYYY-MM-DD, got {report_date!r}")
    entries = _validate_files(files)
    repo_dir = Path(repo_dir)
    if not repo_dir.is_dir():
        raise FigureHostError(f"repo_dir {str(repo_dir)!r} is not a directory")

    git = _Git(repo_dir)
    git.run("rev-parse", "--git-dir")  # fails clearly when repo_dir isn't inside a checkout
    _check_branch_name(git, branch)
    slug = _resolve_slug(git, remote, repo_slug)

    blobs = _hash_files(git, entries)
    commit = _commit_and_push(git, blobs, report_date, branch=branch, remote=remote)
    if commit:
        logger.info("Published %d figure(s) for %s to %s/%s (commit %s)", len(blobs), report_date, remote, branch, commit[:12])
    else:
        logger.info("Figures for %s are already on %s/%s; nothing to commit", report_date, remote, branch)

    urls = {
        paper_url: RAW_URL_TEMPLATE.format(slug=slug, branch=branch, date=report_date, filename=dest)
        for paper_url, _, dest in entries
    }
    if wait_timeout_s > 0:
        wait_until_reachable(list(urls.values()), wait_timeout_s, sleep=sleep, clock=clock)
    return urls


def wait_until_reachable(
    urls: Sequence[str],
    timeout_s: float,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Poll until every URL answers 200 with an ``image/*`` Content-Type, backing off 1, 2, 4, 8, 15, 15... s.

    Raises ``FigureHostError`` naming the URLs that still weren't ready after ``timeout_s``.
    """
    pending = list(dict.fromkeys(urls))
    deadline = clock() + timeout_s
    token = uuid.uuid4().hex[:8]
    delay = POLL_INITIAL_S
    attempt = 0
    while True:
        attempt += 1
        pending = [url for url in pending if not _serves_image(url, f"{token}-{attempt}")]
        if not pending:
            return
        remaining = deadline - clock()
        if remaining <= 0:
            raise FigureHostError(
                f"{len(pending)} figure URL(s) not reachable after {timeout_s:.0f} s: " + ", ".join(pending)
            )
        logger.info("Waiting for %d figure URL(s) to become reachable (retry in %.0f s)", len(pending), min(delay, remaining))
        sleep(min(delay, remaining))
        delay = min(delay * 2, POLL_MAX_INTERVAL_S)


def parse_repo_slug(remote_url: str) -> Optional[str]:
    """``"owner/name"`` from a github.com remote URL (https, ssh:// or scp-like ``git@github.com:o/n.git``), else None."""
    url = (remote_url or "").strip()
    if "://" in url:
        parts = urlsplit(url)
        host, path = (parts.hostname or ""), parts.path
    else:
        m = _SCP_RE.match(url)
        if not m:
            return None
        host, path = m.group("host"), m.group("path")
    host = host.lower()
    if host not in _GITHUB_HOSTS and not host.startswith("github.com-"):  # github.com-<alias>: ssh config aliases
        return None
    path = path.strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    return path if _SLUG_RE.match(path) else None


def resolve_repo_slug(repo_dir: Path, remote: str = "origin", repo_slug: Optional[str] = None) -> str:
    """The ``owner/name`` used in the URLs: ``repo_slug``, else ``$GITHUB_REPOSITORY``, else parsed from the remote URL."""
    return _resolve_slug(_Git(Path(repo_dir)), remote, repo_slug)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


class _Git:
    """Runs git inside ``repo_dir``; every failure becomes a redacted ``FigureHostError``."""

    def __init__(self, repo_dir: Path) -> None:
        self.repo_dir = repo_dir
        self._env = dict(os.environ)
        self._env["GIT_TERMINAL_PROMPT"] = "0"  # fail instead of waiting for a password on a TTY

    def run(
        self,
        *args: str,
        input: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        network: bool = False,
    ) -> str:
        what = f"git {_subcommand(args)}"
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=str(self.repo_dir),
                input=input,
                env={**self._env, **(env or {})},
                check=True,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=NETWORK_GIT_TIMEOUT_S if network else LOCAL_GIT_TIMEOUT_S,
            )
        except FileNotFoundError:
            raise FigureHostError("git is not installed or not on PATH") from None
        except subprocess.TimeoutExpired as exc:
            raise FigureHostError(f"`{what}` timed out after {exc.timeout:.0f} s") from None
        except subprocess.CalledProcessError as exc:
            detail = redact(exc.stderr or exc.stdout or "")
            raise FigureHostError(f"`{what}` failed (exit {exc.returncode}): {detail or 'no output'}") from None
        return proc.stdout


def redact(text: str) -> str:
    """Git output with credentials (URL userinfo, auth headers, GitHub tokens) masked, trimmed to its tail."""
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    text = " ".join(text.split())
    if len(text) > STDERR_SNIPPET_CHARS:
        text = "..." + text[-STDERR_SNIPPET_CHARS:]
    return text


def _subcommand(args: Sequence[str]) -> str:
    """The git subcommand in ``args``, skipping ``-c key=value`` options."""
    skip = False
    for arg in args:
        if skip:
            skip = False
        elif arg == "-c":
            skip = True
        elif not arg.startswith("-"):
            return arg
    return " ".join(args)


def _validate_files(files: Mapping[str, Tuple[Path, str]]) -> List[Tuple[str, Path, str]]:
    entries: List[Tuple[str, Path, str]] = []
    for paper_url, (local_path, dest) in files.items():
        if not isinstance(dest, str) or not _FILENAME_RE.match(dest):
            raise FigureHostError(f"Destination name for {paper_url} must be a plain file name, got {dest!r}")
        if not dest.lower().endswith(IMAGE_EXTENSIONS):
            raise FigureHostError(
                f"Destination name {dest!r} for {paper_url} needs an image extension ({', '.join(IMAGE_EXTENSIONS)}); "
                "raw.githubusercontent.com picks the Content-Type from it"
            )
        path = Path(local_path)
        if not path.is_file():
            raise FigureHostError(f"Figure file for {paper_url} not found: {path}")
        entries.append((paper_url, path.resolve(), dest))
    return entries


def _check_branch_name(git: _Git, branch: str) -> None:
    try:
        git.run("check-ref-format", f"refs/heads/{branch}")
    except FigureHostError:
        raise FigureHostError(f"Invalid branch name {branch!r}") from None


def _resolve_slug(git: _Git, remote: str, repo_slug: Optional[str]) -> str:
    if repo_slug:
        if not _SLUG_RE.match(repo_slug):
            raise FigureHostError(f"repo_slug must look like 'owner/name', got {repo_slug!r}")
        return repo_slug
    env_slug = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if env_slug:
        if not _SLUG_RE.match(env_slug):
            raise FigureHostError(f"GITHUB_REPOSITORY must look like 'owner/name', got {env_slug!r}")
        return env_slug
    remote_url = git.run("remote", "get-url", remote).strip()
    slug = parse_repo_slug(remote_url)
    if not slug:
        # The URL itself is left out of the message: it may embed a token.
        raise FigureHostError(
            f"Can't tell the GitHub repository from remote {remote!r} (not a github.com URL); "
            "pass repo_slug='owner/name' or set GITHUB_REPOSITORY"
        )
    return slug


def _hash_files(git: _Git, entries: Sequence[Tuple[str, Path, str]]) -> Dict[str, str]:
    """Write each image into the object store; ``{dest_filename: blob sha}``."""
    paths = list(dict.fromkeys(path for _, path, _ in entries))
    shas = git.run("hash-object", "-w", "--no-filters", "--", *(str(p) for p in paths)).split()
    if len(shas) != len(paths):
        raise FigureHostError(f"`git hash-object` returned {len(shas)} hashes for {len(paths)} files")
    sha_by_path = dict(zip(paths, shas))
    blobs: Dict[str, str] = {}
    for paper_url, path, dest in entries:
        sha = sha_by_path[path]
        if blobs.get(dest, sha) != sha:
            raise FigureHostError(f"Two different figures would both be published as {dest!r} (one is for {paper_url})")
        blobs[dest] = sha
    return blobs


def _commit_and_push(git: _Git, blobs: Mapping[str, str], report_date: str, *, branch: str, remote: str) -> Optional[str]:
    """Commit ``blobs`` under ``report_date/`` on top of the remote branch and push; None when nothing changed."""
    shallow = git.run("rev-parse", "--is-shallow-repository").strip() == "true"
    identity = _identity_args(git)
    for attempt in range(1, PUSH_ATTEMPTS + 1):
        parent = _fetch_branch_tip(git, remote, branch, shallow=shallow)
        commit = _build_commit(git, parent, blobs, report_date, identity)
        if commit is None:
            return None
        try:
            git.run("push", "--no-verify", "--quiet", remote, f"{commit}:refs/heads/{branch}", network=True)
        except FigureHostError as exc:
            if attempt == PUSH_ATTEMPTS:
                raise FigureHostError(f"Pushing figures to {remote}/{branch} failed after {attempt} attempts: {exc}") from None
            logger.warning("Pushing figures to %s/%s failed (%s); re-fetching and retrying once", remote, branch, exc)
            continue
        return commit
    return None  # unreachable: the last attempt either returns or raises


def _fetch_branch_tip(git: _Git, remote: str, branch: str, *, shallow: bool) -> Optional[str]:
    """Fetch ``remote``'s ``branch`` and return its tip sha, or None when the branch doesn't exist yet."""
    temp_ref = f"{TEMP_REF_PREFIX}{uuid.uuid4().hex}"
    args = ["fetch", "--no-tags", "--quiet"]
    if shallow:
        args.append("--depth=1")
    args += [remote, f"+refs/heads/{branch}:{temp_ref}"]
    try:
        try:
            git.run(*args, network=True)
        except FigureHostError as fetch_error:
            # Tell "no such branch" (create it) from real failures without parsing localized messages.
            try:
                listing = git.run("ls-remote", remote, f"refs/heads/{branch}", network=True)
            except FigureHostError:
                raise FigureHostError(f"Fetching {remote}/{branch} failed: {fetch_error}") from None
            if any(line.split("\t")[-1].strip() == f"refs/heads/{branch}" for line in listing.splitlines()):
                raise FigureHostError(f"Fetching {remote}/{branch} failed: {fetch_error}") from None
            logger.info("Branch %s doesn't exist on %s yet; creating it as an orphan branch", branch, remote)
            return None
        return git.run("rev-parse", "--verify", f"{temp_ref}^{{commit}}").strip()
    finally:
        try:
            git.run("update-ref", "-d", temp_ref)
        except FigureHostError:
            pass  # the ref was never created


def _build_commit(
    git: _Git,
    parent: Optional[str],
    blobs: Mapping[str, str],
    report_date: str,
    identity: Sequence[str],
) -> Optional[str]:
    """Commit ``parent``'s tree plus ``blobs`` (orphan with README.md when no parent); None when the tree is unchanged."""
    lines = []
    if parent is None:
        readme_sha = git.run("hash-object", "-w", "--stdin", input=README_TEXT).strip()
        lines.append(f"100644 blob {readme_sha}\tREADME.md")
    lines.extend(f"100644 blob {sha}\t{report_date}/{dest}" for dest, sha in sorted(blobs.items()))

    tmpdir = tempfile.mkdtemp(prefix="figure-host-")
    try:
        env = {"GIT_INDEX_FILE": os.path.join(tmpdir, "index")}
        no_split = ("-c", "core.splitIndex=false")  # keep shared-index files out of .git
        if parent is not None:
            git.run(*no_split, "read-tree", parent, env=env)
        git.run(
            *no_split, "update-index", "--add", "--replace", "-z", "--index-info",
            input="".join(line + "\0" for line in lines), env=env,
        )
        tree = git.run(*no_split, "write-tree", env=env).strip()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if parent is not None and tree == git.run("rev-parse", f"{parent}^{{tree}}").strip():
        return None
    args = [*identity, "commit-tree", tree]
    if parent is not None:
        args += ["-p", parent]
    args += ["-m", f"figures: {report_date}"]
    return git.run(*args).strip()


def _identity_args(git: _Git) -> List[str]:
    """``-c user.name/user.email`` for the bot unless both are configured (GIT_AUTHOR_*/GIT_COMMITTER_* still win)."""
    configured = []
    for key in ("user.name", "user.email"):
        try:
            configured.append(bool(git.run("config", "--get", key).strip()))
        except FigureHostError:  # exit 1: not set
            configured.append(False)
    if all(configured):
        return []
    return ["-c", f"user.name={BOT_NAME}", "-c", f"user.email={BOT_EMAIL}"]


def _serves_image(url: str, nonce: str) -> bool:
    """One HEAD check. The query string bypasses the CDN cache (raw.githubusercontent.com caches 404s for ~5 min)."""
    try:
        resp = requests.head(f"{url}?figure_host_check={nonce}", allow_redirects=True, timeout=HEAD_TIMEOUT_S)
    except requests.RequestException as exc:
        logger.debug("HEAD %s failed: %s", url, exc)
        return False
    content_type = (resp.headers.get("Content-Type") or "").lower()
    ready = resp.status_code == 200 and content_type.startswith("image/")
    if not ready:
        logger.debug("HEAD %s -> %s %s", url, resp.status_code, content_type or "(no Content-Type)")
    return ready
