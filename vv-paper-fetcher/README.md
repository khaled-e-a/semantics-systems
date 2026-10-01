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

## Substack draft post

**First-time setup:** follow the checklist in [SUBSTACK_SETUP.md](SUBSTACK_SETUP.md).

`write_post.py` turns the week's report into a Substack post that follows
`WRITING_SUBSTACK_POST.md` (strict VV/UQ filter, ASD-STE100 English, exact
links, no author names), puts each kept paper's main figure (with a credit
caption) above its paragraph, and saves it as a **draft** on Substack. It
never publishes: you get an email with a link to the draft, review it, and
publish by hand. The post is also committed as
`reports/YYYY-MM-DD-substack-post-ste100.md`; the figure files are not.

**Trigger.** `.github/workflows/substack-draft.yml` (repo root) runs after
each successful "Weekly VV/UQ/Evals Paper Digest" run. You can also run it by
hand (Actions → Substack Draft Post → Run workflow) with:
- `report_date` — which report to use (default: the newest one);
- `force` — redo a date that already has a draft. This **updates** the same
  draft instead of creating a second one (a new draft is created if you
  deleted the old one);
- `smoke_test` — only check that the runner can reach Substack (see below).

**What you get by email:**

| Outcome | Email |
|---|---|
| Draft created | "Substack draft ready: …" with the editor link, kept/total papers, each paper's contribution type and figure, figures that are not openly licensed, any checks that still fail, and the dropped papers with reasons. |
| Substack failed (expired cookie, Cloudflare block, not configured) | "Substack draft NOT created — DATE" with the reason, how to fix it, and the whole post as HTML with the figures attached, so you can still publish by hand. |
| Quiet week, or no paper passed the filter | "No Substack post this week — DATE" with the reasons papers were dropped. |
| The LLM never returned a usable post | "Substack post NOT written — DATE" with the reason, the selected papers, and their figures attached. The workflow run is also marked failed. |
| Draft already exists for the date | Nothing: the run is skipped (use `force`). |
| Unexpected error | The workflow fails and GitHub sends its usual failure email. |

### Secrets and variables

| Name | Kind | Required | What |
|---|---|---|---|
| `SUBSTACK_PUBLICATION_URL` | secret | yes | `https://<name>.substack.com` — the substack.com address, not a custom domain. |
| `SUBSTACK_COOKIE` | secret | yes | The full `Cookie` request header of a logged-in substack.com session (see below). |
| `SUBSTACK_WRITER_MODEL` | secret | no | OpenRouter model for writing the post. Default: `OPENROUTER_MODEL`. |
| `SUBSTACK_PROXY` | secret | no | Proxy URL for Substack calls only, e.g. `socks5h://user:pass@host:port` or `http://host:port`. |
| `SUBSTACK_USE_WARP` | variable | no | Set to `true` to route Substack calls through Cloudflare WARP (see below). |
| `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `RESEND_API_KEY`, `REPORT_EMAIL_FROM`, `REPORT_EMAIL_TO` | secret | yes | Same values as the digest. |

### Getting `SUBSTACK_COOKIE`

1. Sign in to substack.com in your browser.
2. Open DevTools (F12 or ⌥⌘I) → **Network** tab, then reload the page.
3. Click any request to `substack.com` (for example `self` or `subscriptions`).
4. Under **Request Headers**, find `Cookie`, and copy its whole value
   (it contains `substack.sid=…` among other cookies).
5. Save it as the repo secret `SUBSTACK_COOKIE` (Settings → Secrets and
   variables → Actions).

The cookie expires after a while, and **signing out of Substack invalidates
it**, so don't sign out of that browser session. When the email says the
draft was not created because of the cookie, copy it again and update the
secret.

### Smoke test, Cloudflare, and WARP

Substack sits behind Cloudflare, which reportedly blocks GitHub-hosted
runners' datacenter IPs even with a valid cookie. Before relying on the
weekly run, run the workflow by hand with `smoke_test` checked: it fetches
your profile, uploads a tiny image, creates a draft, and deletes it. Locally:
`python -m src.substack_post.substack_client --smoke-test`.

If the smoke test is blocked, set the repo **variable** `SUBSTACK_USE_WARP`
to `true` and run it again: the workflow then installs Cloudflare WARP in
local proxy mode and sends only the Substack calls through it
(`SUBSTACK_PROXY=socks5h://127.0.0.1:40000`; this takes precedence over the
`SUBSTACK_PROXY` secret). If WARP is blocked too, put a residential proxy in
the `SUBSTACK_PROXY` secret (and unset `SUBSTACK_USE_WARP`), or use a
self-hosted runner.

### Running locally

```bash
# Print the post (figures as local file paths); no Substack, email, or file writes
python write_post.py --dry-run --verbose
python write_post.py --date 2026-09-28 --dry-run --verbose

# Write the post file and email it with the figures attached, without Substack
python write_post.py --skip-substack

# Re-do a date that already has a draft: updates that same draft
python write_post.py --date 2026-09-28 --force
```

`--dry-run` also runs for dates that already have a draft. Already-drafted
dates are tracked in `state/substack_drafts.json`.

### Style example

The writer imitates one earlier post. Set `substack_post.style_example` in
`config.yaml` to a post **you have reviewed and approved** (default:
`reports/2026-09-07-substack-post-ste100.md`), and update it when you approve
a better one. If the key is unset, the newest
`reports/*-substack-post-ste100.md` is used instead, but that soon becomes the
pipeline's own unreviewed output, and the style would drift.

## Testing

```bash
pytest
```

Unit tests cover the two riskiest failure paths: malformed LLM JSON
(retry-then-skip) and a source outage (`collect_all()` degrading instead of
raising).
