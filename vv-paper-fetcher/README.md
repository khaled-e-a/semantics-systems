# vv-paper-fetcher

Weekly digest of new papers on **verification, validation, uncertainty
quantification, and evals** for AI agents, LLMs, symbolic AI, scientific AI,
and formal AI — collected from arXiv, Hugging Face Papers, OpenReview
(NeurIPS/ICML/ICLR), and DBLP (CAV/TACAS/FM/POPL/CADE/LICS/VMCAI), triaged and
summarized with an LLM via OpenRouter, and ranked primarily by author
reputation (h-index/citations via Semantic Scholar) rather than the new
paper's own (near-zero) citation count.

Runs every Monday via GitHub Actions (`.github/workflows/weekly-papers.yml`
at the repo root), writes `reports/YYYY-MM-DD.md`, and emails a digest via
Resend.

## Setup

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in:
   - `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` — required, any OpenRouter-hosted model
   - `RESEND_API_KEY`, `REPORT_EMAIL_FROM`, `REPORT_EMAIL_TO` — required. Resend's
     sandbox sender (`onboarding@resend.dev`) only delivers to your own
     account email; verify a custom domain in Resend to send elsewhere.
   - `SEMANTIC_SCHOLAR_API_KEY` — optional but recommended (free), raises you
     off the shared unauthenticated rate limit.
3. For the GitHub Actions workflow, add the same values as repo secrets.

## Usage

```bash
# Full pipeline, no email/state writes, prints report to stdout
python main.py --dry-run --verbose

# Isolate one collector during development
python main.py --dry-run --sources arxiv

# Backfill / test against a specific past week
python main.py --dry-run --since 2026-08-01 --until 2026-08-08
```

## How it works

See `config.yaml` for all tunable settings (source categories/venues,
keyword pre-filter terms, LLM triage batch size, scoring weights). Nothing
in `config.yaml` requires a code change to adjust.

Pipeline: collect (per-source, fault-isolated) → in-run dedup/merge →
keyword pre-filter → cross-run dedup (`state/seen_papers.json`) → LLM triage
(OpenRouter, title+abstract only, batched) → reputation lookup (Semantic
Scholar, post-triage only) → composite scoring/ranking → render (Markdown +
HTML) → write report → send email → update state.

A source being down, an LLM batch failing to parse, or Semantic Scholar
rate-limiting mid-run all degrade gracefully — the run still produces a
report from whatever succeeded. A week with zero relevant papers is a valid
"quiet week" outcome, not a failure.

## Testing

```bash
pytest
```

Unit tests cover the two riskiest failure paths: malformed LLM JSON
(retry-then-skip) and a source outage (`collect_all()` degrading instead of
raising).
