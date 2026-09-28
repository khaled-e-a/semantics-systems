"""Sanity checks for .github/workflows/substack-draft.yml (at the repo root)."""
import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
WORKFLOW = WORKFLOWS / "substack-draft.yml"


@pytest.fixture(scope="module")
def wf():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _triggers(wf):
    return wf.get("on", wf.get(True))  # YAML 1.1 parses a bare `on` key as True


def _steps(wf):
    return wf["jobs"]["write-draft"]["steps"]


def _step(wf, name_part):
    (step,) = [s for s in _steps(wf) if name_part in s.get("name", "")]
    return step


def test_triggers_follow_the_digest_workflow_by_name(wf):
    triggers = _triggers(wf)
    digest = yaml.safe_load((WORKFLOWS / "weekly-papers.yml").read_text(encoding="utf-8"))
    assert triggers["workflow_run"]["workflows"] == [digest["name"]]
    assert triggers["workflow_run"]["types"] == ["completed"]

    inputs = triggers["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"report_date", "force", "smoke_test"}
    assert inputs["report_date"]["type"] == "string" and inputs["report_date"]["required"] is False
    for name in ("force", "smoke_test"):
        assert inputs[name]["type"] == "boolean" and inputs[name]["default"] is False


def test_job_guards_permissions_and_concurrency(wf):
    job = wf["jobs"]["write-draft"]
    assert "github.event_name == 'workflow_dispatch'" in job["if"]
    assert "github.event.workflow_run.conclusion == 'success'" in job["if"]
    assert wf["permissions"] == {"contents": "write"}
    assert wf["concurrency"]["group"] == "substack-draft"
    assert wf["defaults"]["run"]["working-directory"] == "vv-paper-fetcher"
    checkout = _steps(wf)[0]
    assert checkout["uses"].startswith("actions/checkout@") and checkout["with"]["ref"] == "main"


def test_warp_smoke_main_and_commit_steps(wf):
    warp = _step(wf, "WARP")
    assert warp["if"] == "vars.SUBSTACK_USE_WARP == 'true'"
    assert "pkg.cloudflareclient.com" in warp["run"]
    assert "mode proxy" in warp["run"]
    assert 'SUBSTACK_PROXY=socks5h://127.0.0.1:40000" >> "$GITHUB_ENV"' in warp["run"]

    smoke = _step(wf, "smoke test")
    assert smoke["if"] == "inputs.smoke_test"
    assert "python -m src.substack_post.substack_client --smoke-test" in smoke["run"]

    main = _step(wf, "Write post")
    assert main["if"] == "${{ !inputs.smoke_test }}"
    assert "python write_post.py" in main["run"]
    for var in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "SUBSTACK_WRITER_MODEL",
        "RESEND_API_KEY",
        "REPORT_EMAIL_TO",
        "REPORT_EMAIL_FROM",
        "SUBSTACK_PUBLICATION_URL",
        "SUBSTACK_COOKIE",
    ):
        assert main["env"][var] == f"${{{{ secrets.{var} }}}}"
    # The WARP value from $GITHUB_ENV must not be shadowed by a step-level env entry.
    assert "SUBSTACK_PROXY" not in main["env"] and "SUBSTACK_PROXY" not in smoke["env"]
    assert main["env"]["SUBSTACK_PROXY_SECRET"] == "${{ secrets.SUBSTACK_PROXY }}"

    commit = _step(wf, "Commit")
    assert "!inputs.smoke_test" in commit["if"]
    assert "state/substack_drafts.json" in commit["run"]
    assert "git pull --rebase" in commit["run"] and "git push" in commit["run"]


def test_no_expressions_inlined_into_scripts(wf):
    """Inputs must reach scripts via env: — `${{ }}` inside run: is an injection risk."""
    for step in _steps(wf):
        assert not re.search(r"\$\{\{", step.get("run", "")), step.get("name")
