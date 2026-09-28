import pytest

from src import config


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)


def test_load_env_vars_returns_required_and_set_optional(monkeypatch):
    monkeypatch.setenv("REQ_A", "a")
    monkeypatch.setenv("OPT_SET", "o")
    monkeypatch.setenv("OPT_EMPTY", "")
    monkeypatch.delenv("OPT_UNSET", raising=False)

    env = config.load_env_vars(["REQ_A"], ["OPT_SET", "OPT_EMPTY", "OPT_UNSET"])
    assert env == {"REQ_A": "a", "OPT_SET": "o"}


def test_load_env_vars_fails_fast_on_missing(monkeypatch, capsys):
    monkeypatch.setenv("REQ_A", "a")
    monkeypatch.delenv("REQ_B", raising=False)
    monkeypatch.setenv("REQ_C", "")

    with pytest.raises(SystemExit) as exc:
        config.load_env_vars(["REQ_A", "REQ_B", "REQ_C"], [])
    assert exc.value.code == 1
    assert "Missing required environment variables: REQ_B, REQ_C" in capsys.readouterr().err
