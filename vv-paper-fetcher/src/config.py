"""Load and validate config.yaml + required environment variables."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

REQUIRED_ENV_VARS = [
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "RESEND_API_KEY",
    "REPORT_EMAIL_TO",
    "REPORT_EMAIL_FROM",
]
# Recommended, not required — the pipeline degrades to a lower rate limit without it.
OPTIONAL_ENV_VARS = ["SEMANTIC_SCHOLAR_API_KEY", "OPENREVIEW_USERNAME", "OPENREVIEW_PASSWORD"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Path | None = None) -> Dict[str, Any]:
    path = config_path or (PROJECT_ROOT / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env(dry_run: bool = False) -> Dict[str, str]:
    """Load .env and fail fast (before any network calls) if required vars are missing.

    In --dry-run mode email/state are not written, so email vars are not required.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    required = list(REQUIRED_ENV_VARS)
    if dry_run:
        required = [v for v in required if v not in {"RESEND_API_KEY", "REPORT_EMAIL_TO", "REPORT_EMAIL_FROM"}]

    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}", file=sys.stderr)
        print("Copy .env.example to .env and fill them in.", file=sys.stderr)
        sys.exit(1)

    env = {v: os.environ[v] for v in required}
    for v in OPTIONAL_ENV_VARS + ["RESEND_API_KEY", "REPORT_EMAIL_TO", "REPORT_EMAIL_FROM"]:
        if os.environ.get(v):
            env[v] = os.environ[v]
    return env
