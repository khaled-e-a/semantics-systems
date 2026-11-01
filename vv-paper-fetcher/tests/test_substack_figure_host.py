"""figure_host against real git: a bare repo in tmp_path plays the GitHub remote."""
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import requests

from src.substack_post import figure_host
from src.substack_post.figure_host import (
    BOT_EMAIL,
    BOT_NAME,
    FigureHostError,
    parse_repo_slug,
    publish_figures,
    redact,
    resolve_repo_slug,
    wait_until_reachable,
)

SLUG = "khaled-e-a/semantics-systems"
PNG_A = b"\x89PNG\r\n\x1a\n" + b"figure-a" * 10
PNG_B = b"\x89PNG\r\n\x1a\n" + b"figure-b" * 10
PNG_C = b"\x89PNG\r\n\x1a\n" + b"figure-c" * 10


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True).stdout


def write_png(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def snapshot(work: Path) -> Tuple[str, str, str, str]:
    """Everything publish_figures must leave alone: status, staged diff, HEAD, current branch."""
    return (
        git(work, "status", "--porcelain", "--untracked-files=all"),
        git(work, "diff", "--cached"),
        git(work, "rev-parse", "HEAD").strip(),
        git(work, "symbolic-ref", "HEAD").strip(),
    )


def remote_files(remote: Path, branch: str = "figures") -> List[str]:
    return sorted(git(remote, "ls-tree", "-r", "--name-only", branch).split())


def remote_count(remote: Path, branch: str = "figures") -> int:
    return int(git(remote, "rev-list", "--count", branch).strip())


def remote_blob(remote: Path, path: str, branch: str = "figures") -> bytes:
    return subprocess.run(
        ["git", "cat-file", "blob", f"{branch}:{path}"], cwd=str(remote), check=True, capture_output=True
    ).stdout


def publish(work: Path, files: Dict[str, Tuple[Path, str]], date: str, **kwargs) -> Dict[str, str]:
    kwargs.setdefault("repo_slug", SLUG)
    kwargs.setdefault("wait_timeout_s", 0)
    return publish_figures(files, date, work / "vv-paper-fetcher", **kwargs)


class FakeResponse:
    def __init__(self, status_code: int, content_type: str = "") -> None:
        self.status_code = status_code
        self.headers = {"Content-Type": content_type} if content_type else {}


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: List[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def hermetic_git(tmp_path, monkeypatch):
    """No user/system git config (signing, hooks, identity) and no real HTTP."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "empty-gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for var in ("GITHUB_REPOSITORY", "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
        monkeypatch.delenv(var, raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("unexpected HTTP request")

    monkeypatch.setattr(figure_host.requests, "head", no_network)


@pytest.fixture
def remote(tmp_path) -> Path:
    path = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", "-q", str(path))
    return path


@pytest.fixture
def work(tmp_path, remote) -> Path:
    """A clone with one commit on main, a staged change and an untracked file (no user identity configured)."""
    path = tmp_path / "work"
    git(tmp_path, "init", "-q", str(path))
    git(path, "checkout", "-q", "-b", "main")
    (path / "README.md").write_text("main readme\n")
    (path / "vv-paper-fetcher").mkdir()
    (path / "vv-paper-fetcher" / "write_post.py").write_text("print('hi')\n")
    git(path, "add", ".")
    git(path, "-c", "user.name=Dev", "-c", "user.email=dev@example.com", "commit", "-q", "-m", "init")
    git(path, "remote", "add", "origin", str(remote))
    git(path, "push", "-q", "origin", "main")
    (path / "README.md").write_text("main readme, edited and staged\n")
    git(path, "add", "README.md")
    (path / "untracked.txt").write_text("scratch\n")
    return path


@pytest.fixture
def figs(tmp_path) -> Dict[str, Path]:
    d = tmp_path / "figs"
    return {
        "a": write_png(d / "fig-2610.00001.png", PNG_A),
        "b": write_png(d / "fig-2610.00002.png", PNG_B),
    }


# ---------------------------------------------------------------------------
# publishing
# ---------------------------------------------------------------------------


def test_first_publish_creates_orphan_branch_and_leaves_checkout_alone(work, remote, figs):
    before = snapshot(work)
    files = {
        "https://arxiv.org/abs/2610.00001": (figs["a"], "fig-2610.00001.png"),
        "https://arxiv.org/abs/2610.00002": (figs["b"], "fig-2610.00002.png"),
    }

    urls = publish(work, files, "2026-10-26")

    assert urls == {
        "https://arxiv.org/abs/2610.00001": f"https://raw.githubusercontent.com/{SLUG}/figures/2026-10-26/fig-2610.00001.png",
        "https://arxiv.org/abs/2610.00002": f"https://raw.githubusercontent.com/{SLUG}/figures/2026-10-26/fig-2610.00002.png",
    }
    assert remote_files(remote) == ["2026-10-26/fig-2610.00001.png", "2026-10-26/fig-2610.00002.png", "README.md"]
    assert remote_count(remote) == 1
    assert git(remote, "rev-list", "--max-parents=0", "figures") == git(remote, "rev-parse", "figures")  # orphan
    assert git(remote, "log", "-1", "--format=%s", "figures").strip() == "figures: 2026-10-26"
    assert remote_blob(remote, "2026-10-26/fig-2610.00001.png") == PNG_A
    readme = remote_blob(remote, "README.md").decode()
    assert "Figures for the weekly Substack posts, published by write_post.py; one folder per report date" in readme
    # main's history is untouched; the caller's checkout is exactly as it was
    assert git(remote, "rev-list", "--count", "main").strip() == "1"
    assert snapshot(work) == before
    assert (work / "README.md").read_text() == "main readme, edited and staged\n"
    assert git(work, "for-each-ref", "refs/figure-host/") == ""  # temp fetch refs cleaned up


def test_second_date_keeps_first_dates_files(work, remote, figs):
    publish(work, {"p1": (figs["a"], "fig-1.png")}, "2026-10-19")
    publish(work, {"p2": (figs["b"], "fig-2.png")}, "2026-10-26")

    assert remote_files(remote) == ["2026-10-19/fig-1.png", "2026-10-26/fig-2.png", "README.md"]
    assert remote_count(remote) == 2
    assert git(remote, "log", "-1", "--format=%s", "figures").strip() == "figures: 2026-10-26"


def test_same_date_identical_bytes_makes_no_new_commit(work, remote, figs):
    first = publish(work, {"p1": (figs["a"], "fig-1.png")}, "2026-10-26")
    tip = git(remote, "rev-parse", "figures")

    again = publish(work, {"p1": (figs["a"], "fig-1.png")}, "2026-10-26")

    assert again == first
    assert git(remote, "rev-parse", "figures") == tip


def test_same_date_changed_bytes_overwrites_with_one_new_commit(work, remote, figs):
    publish(work, {"p1": (figs["a"], "fig-1.png")}, "2026-10-26")
    figs["a"].write_bytes(PNG_C)

    publish(work, {"p1": (figs["a"], "fig-1.png")}, "2026-10-26")

    assert remote_count(remote) == 2
    assert remote_blob(remote, "2026-10-26/fig-1.png") == PNG_C
    assert remote_files(remote) == ["2026-10-26/fig-1.png", "README.md"]


def test_works_from_shallow_main_only_clone_like_actions_checkout(tmp_path, work, remote, figs):
    """actions/checkout@v4: fetch-depth 1, only main; the module must fetch figures itself."""

    def shallow_clone(name: str) -> Path:
        dest = tmp_path / name
        git(tmp_path, "clone", "-q", "--depth=1", "--single-branch", "--branch", "main", f"file://{remote}", str(dest))
        assert git(dest, "rev-parse", "--is-shallow-repository").strip() == "true"
        return dest

    ci1 = shallow_clone("ci1")
    publish(ci1, {"p1": (figs["a"], "fig-1.png")}, "2026-10-19")
    ci2 = shallow_clone("ci2")  # created after figures exists, still main only
    before = snapshot(ci2)

    publish(ci2, {"p2": (figs["b"], "fig-2.png")}, "2026-10-26")

    assert remote_files(remote) == ["2026-10-19/fig-1.png", "2026-10-26/fig-2.png", "README.md"]
    assert remote_count(remote) == 2
    assert snapshot(ci2) == before


def test_full_clone_is_not_made_shallow(work, remote, figs):
    publish(work, {"p1": (figs["a"], "fig-1.png")}, "2026-10-19")
    publish(work, {"p2": (figs["b"], "fig-2.png")}, "2026-10-26")

    assert git(work, "rev-parse", "--is-shallow-repository").strip() == "false"


def test_commit_identity_falls_back_to_bot(work, remote, figs):
    publish(work, {"p1": (figs["a"], "fig-1.png")}, "2026-10-26")

    assert git(remote, "log", "-1", "--format=%an <%ae>|%cn <%ce>", "figures").strip() == (
        f"{BOT_NAME} <{BOT_EMAIL}>|{BOT_NAME} <{BOT_EMAIL}>"
    )


def test_commit_identity_uses_configured_user(work, remote, figs):
    git(work, "config", "user.name", "Khaled")
    git(work, "config", "user.email", "khaled@example.com")

    publish(work, {"p1": (figs["a"], "fig-1.png")}, "2026-10-26")

    assert git(remote, "log", "-1", "--format=%an <%ae>", "figures").strip() == "Khaled <khaled@example.com>"


def test_push_conflict_refetches_and_retries_once(tmp_path, work, remote, figs, monkeypatch):
    publish(work, {"p1": (figs["a"], "fig-1.png")}, "2026-10-19")

    # Another runner pushes to figures between our fetch and our push.
    other = tmp_path / "other"
    git(tmp_path, "clone", "-q", "--branch", "figures", str(remote), str(other))
    write_png(other / "2026-10-20" / "theirs.png", PNG_C)
    git(other, "add", ".")
    git(other, "-c", "user.name=Other", "-c", "user.email=o@example.com", "commit", "-q", "-m", "figures: 2026-10-20")

    real_run = figure_host._Git.run
    pushes: List[str] = []

    def run(self, *args, **kwargs):
        if "push" in args:
            if not pushes:
                git(other, "push", "-q", "origin", "figures")
            pushes.append(args[-1])
        return real_run(self, *args, **kwargs)

    monkeypatch.setattr(figure_host._Git, "run", run)

    publish(work, {"p2": (figs["b"], "fig-2.png")}, "2026-10-26")

    assert len(pushes) == 2
    assert remote_files(remote) == ["2026-10-19/fig-1.png", "2026-10-20/theirs.png", "2026-10-26/fig-2.png", "README.md"]
    assert remote_count(remote) == 3


def test_push_rejected_by_remote_raises_and_cleans_up(tmp_path, work, remote, figs, monkeypatch):
    hook = remote / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\necho 'push refused by test hook' >&2\nexit 1\n")
    hook.chmod(0o755)
    real_mkdtemp = tempfile.mkdtemp
    made: List[str] = []

    def mkdtemp(*args, **kwargs):
        made.append(real_mkdtemp(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(figure_host.tempfile, "mkdtemp", mkdtemp)
    before = snapshot(work)

    with pytest.raises(FigureHostError, match="Pushing figures to origin/figures failed after 2 attempts") as excinfo:
        publish(work, {"p1": (figs["a"], "fig-1.png")}, "2026-10-26")

    assert "push refused by test hook" in str(excinfo.value)
    assert git(remote, "branch", "--list", "figures") == ""
    assert snapshot(work) == before
    assert len(made) == 2 and not any(Path(d).exists() for d in made)  # one temp index per attempt, both removed
    assert git(work, "for-each-ref", "refs/figure-host/") == ""


def test_missing_remote_raises(work, remote, figs):
    shutil.rmtree(remote)
    before = snapshot(work)

    with pytest.raises(FigureHostError, match="origin/figures"):
        publish(work, {"p1": (figs["a"], "fig-1.png")}, "2026-10-26")

    assert snapshot(work) == before


def test_publish_waits_for_every_url(work, remote, figs, monkeypatch):
    calls: List[str] = []

    def head(url, allow_redirects=False, timeout=None):
        assert allow_redirects is True
        calls.append(url)
        return FakeResponse(200, "image/png")

    monkeypatch.setattr(figure_host.requests, "head", head)

    urls = publish(work, {"p1": (figs["a"], "fig-1.png"), "p2": (figs["b"], "fig-2.png")}, "2026-10-26", wait_timeout_s=60)

    assert len(calls) == 2
    assert sorted(c.split("?")[0] for c in calls) == sorted(urls.values())


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def test_empty_files_returns_empty_without_git(tmp_path):
    assert publish_figures({}, "2026-10-26", tmp_path / "not-a-repo") == {}


@pytest.mark.parametrize("date", ["2026-10-6", "26-10-2026", "", "2026-10-26/x"])
def test_bad_report_date_raises(work, figs, date):
    with pytest.raises(FigureHostError, match="YYYY-MM-DD"):
        publish(work, {"p1": (figs["a"], "fig-1.png")}, date)


@pytest.mark.parametrize("dest", ["../fig.png", "a/fig.png", ".hidden.png", "fig.txt", "fig"])
def test_bad_destination_name_raises(work, figs, dest):
    with pytest.raises(FigureHostError):
        publish(work, {"p1": (figs["a"], dest)}, "2026-10-26")


def test_missing_local_file_raises(work, tmp_path):
    with pytest.raises(FigureHostError, match="not found"):
        publish(work, {"p1": (tmp_path / "nope.png", "fig-1.png")}, "2026-10-26")


def test_two_different_files_with_same_destination_raise(work, figs):
    with pytest.raises(FigureHostError, match="fig-1.png"):
        publish(work, {"p1": (figs["a"], "fig-1.png"), "p2": (figs["b"], "fig-1.png")}, "2026-10-26")


def test_repo_dir_outside_a_checkout_raises(tmp_path, figs):
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(FigureHostError, match="rev-parse"):
        publish_figures({"p1": (figs["a"], "fig-1.png")}, "2026-10-26", outside, repo_slug=SLUG, wait_timeout_s=0)


# ---------------------------------------------------------------------------
# repo slug
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/khaled-e-a/semantics-systems.git",
        "https://github.com/khaled-e-a/semantics-systems",
        "https://github.com/khaled-e-a/semantics-systems/",
        "https://x-access-token:ghs_abc@github.com/khaled-e-a/semantics-systems.git",
        "git@github.com:khaled-e-a/semantics-systems.git",
        "git@github.com:khaled-e-a/semantics-systems",
        "ssh://git@github.com/khaled-e-a/semantics-systems.git",
        "ssh://git@ssh.github.com:443/khaled-e-a/semantics-systems.git",
        "git@github.com-personal:khaled-e-a/semantics-systems.git",
    ],
)
def test_parse_repo_slug_github_forms(url):
    assert parse_repo_slug(url) == SLUG


@pytest.mark.parametrize(
    "url",
    ["/tmp/remote.git", "file:///tmp/remote.git", "https://gitlab.com/a/b.git", "https://github.com/only-owner", ""],
)
def test_parse_repo_slug_rejects_non_github(url):
    assert parse_repo_slug(url) is None


def test_slug_precedence_explicit_then_env_then_remote(work, monkeypatch):
    git(work, "remote", "add", "gh", "git@github.com:from/remote.git")
    assert resolve_repo_slug(work, "gh") == "from/remote"
    monkeypatch.setenv("GITHUB_REPOSITORY", "from/env")
    assert resolve_repo_slug(work, "gh") == "from/env"
    assert resolve_repo_slug(work, "gh", "explicit/slug") == "explicit/slug"


def test_publish_uses_github_repository_env(work, figs, monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/from-env")
    urls = publish_figures({"p1": (figs["a"], "fig-1.png")}, "2026-10-26", work, wait_timeout_s=0)
    assert urls == {"p1": "https://raw.githubusercontent.com/owner/from-env/figures/2026-10-26/fig-1.png"}


def test_unparseable_remote_raises_without_leaking_url(work):
    git(work, "remote", "add", "gl", "https://x-access-token:SECRET123@gitlab.example.com/o/r.git")
    with pytest.raises(FigureHostError, match="repo_slug") as excinfo:
        resolve_repo_slug(work, "gl")
    assert "SECRET123" not in str(excinfo.value)


def test_redact_masks_credentials():
    text = (
        "fatal: unable to access 'https://x-access-token:ghs_abcdefghijklmnopqrstuvwxyz0123@github.com/o/r.git/'\n"
        "http.https://github.com/.extraheader AUTHORIZATION: basic eC1hY2Nlc3MtdG9rZW46Z2hz\n"
        "token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345 leaked"
    )
    out = redact(text)
    for secret in ("ghs_abc", "eC1hY2Nlc3MtdG9rZW46Z2hz", "ghp_ABCDEFGHIJ", "x-access-token"):
        assert secret not in out


# ---------------------------------------------------------------------------
# reachability
# ---------------------------------------------------------------------------


def test_wait_until_reachable_retries_until_image(monkeypatch):
    responses = [FakeResponse(404, "text/plain"), requests.ConnectionError("boom"), FakeResponse(200, "text/html"),
                 FakeResponse(200, "image/png")]
    calls: List[str] = []

    def head(url, allow_redirects=False, timeout=None):
        calls.append(url)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(figure_host.requests, "head", head)
    clock = FakeClock()

    wait_until_reachable(["https://raw.example/x.png"], 120, sleep=clock.sleep, clock=clock)

    assert len(calls) == 4
    assert clock.sleeps == [1.0, 2.0, 4.0]
    assert all(c.startswith("https://raw.example/x.png?") for c in calls)
    assert len(set(calls)) == 4  # a fresh cache-busting query each time


def test_wait_until_reachable_times_out_listing_pending_urls(monkeypatch):
    ready = "https://raw.example/ready.png"
    stuck = "https://raw.example/stuck.png"

    def head(url, allow_redirects=False, timeout=None):
        return FakeResponse(200, "image/png") if url.startswith(ready) else FakeResponse(404, "text/plain")

    monkeypatch.setattr(figure_host.requests, "head", head)
    clock = FakeClock()

    with pytest.raises(FigureHostError) as excinfo:
        wait_until_reachable([ready, stuck], 30, sleep=clock.sleep, clock=clock)

    assert stuck in str(excinfo.value)
    assert ready not in str(excinfo.value)
    assert sum(clock.sleeps) == pytest.approx(30)
    assert max(clock.sleeps) <= figure_host.POLL_MAX_INTERVAL_S


def test_publish_raises_when_urls_never_become_reachable(work, remote, figs, monkeypatch):
    monkeypatch.setattr(figure_host.requests, "head", lambda *a, **k: FakeResponse(404, "text/plain"))
    clock = FakeClock()

    with pytest.raises(FigureHostError, match="not reachable"):
        publish(work, {"p1": (figs["a"], "fig-1.png")}, "2026-10-26", wait_timeout_s=10, sleep=clock.sleep, clock=clock)

    assert remote_files(remote) == ["2026-10-26/fig-1.png", "README.md"]  # the push itself went through
