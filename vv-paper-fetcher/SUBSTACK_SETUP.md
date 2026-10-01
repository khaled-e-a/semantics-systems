# Substack draft: setup checklist

What you need to do to get the weekly Substack draft working. How the
pipeline works, and every option, is in [README.md](README.md) under
"Substack draft post".

**Already done:** the repo secrets `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`,
`RESEND_API_KEY`, `REPORT_EMAIL_FROM`, and `REPORT_EMAIL_TO` are set. The
digest uses them too.

## One-time setup (about 10 minutes)

### 1. Find your Substack address

You need the `https://<name>.substack.com` form, even if your publication has
a custom domain. It is in the Substack dashboard under **Settings**.

### 2. Copy your Substack cookie (Chrome)

1. While signed in, open `https://<name>.substack.com/publish/home`.
2. Press **Cmd+Option+I** to open DevTools, then open the **Network** tab.
3. Press **Cmd+R** to reload. Type `api/v1` in the filter box.
4. Click any request to `<name>.substack.com`. In **Headers**, scroll to
   **Request Headers** and find `cookie`.
5. Right-click its value and choose **Copy value**. It is a long string that
   contains `substack.sid=` or `connect.sid=`.

The cookie works like your password: keep it only in the GitHub secret, never
in a file or a commit. Do not sign out of Substack in that browser, because
signing out makes the cookie invalid.

### 3. Note when the cookie expires

1. In the same DevTools window, open **Application → Cookies**, then click
   the `https://<name>.substack.com` entry (or `https://substack.com` if it
   is listed).
2. Find `substack.sid` (or `connect.sid`) and read the **Expires** column.
3. Put a calendar reminder a few days before that date to do step 2 again.

### 4. Add three repo secrets

1. Open https://github.com/khaled-e-a/semantics-systems/settings/secrets/actions
2. For each row, click **New repository secret**, enter the name and the
   value, and click **Add secret**.

| Name | Value |
|---|---|
| `SUBSTACK_PUBLICATION_URL` | `https://<name>.substack.com` (no slash at the end) |
| `SUBSTACK_COOKIE` | the cookie value from step 2 |
| `SUBSTACK_WRITER_MODEL` | `anthropic/claude-opus-5.5` (best writing) or `anthropic/claude-sonnet-5.5` (cheaper) |

`SUBSTACK_WRITER_MODEL` is optional, but recommended: the default triage model
writes dense posts with unexplained jargon.

### 5. Run the smoke test

The smoke test checks that GitHub can reach your Substack. It creates a test
draft and deletes it.

1. Open https://github.com/khaled-e-a/semantics-systems/actions/workflows/substack-draft.yml
2. Click **Run workflow** (on the right). Keep the branch `main`, check
   **"Only check Substack connectivity …"**, and click **Run workflow**.
3. After a minute or two, refresh the page and open the run.

- **Green:** go to step 6.
- **Red, and the log says `Hint: refresh SUBSTACK_COOKIE`:** the cookie is
  wrong. Do step 2 again, then update the secret (secrets page → pencil icon
  next to `SUBSTACK_COOKIE` → paste → **Update secret**). Run the smoke test
  again.
- **Red, and the log says `Cloudflare is blocking this IP`:** open
  https://github.com/khaled-e-a/semantics-systems/settings/variables/actions
  (the **Variables** tab, not Secrets). Click **New repository variable**, set
  name `SUBSTACK_USE_WARP` and value `true`, and click **Add variable**. Run
  the smoke test again.
- **Still blocked with WARP:** the remaining choices are a paid residential
  proxy (put its URL in a `SUBSTACK_PROXY` secret and remove
  `SUBSTACK_USE_WARP`), or running the Substack step on your own Mac. Ask
  Claude to set up whichever you choose.

### 6. Do the first real run

1. Click **Run workflow** again. Leave the smoke-test box unchecked, and set
   the report date to `2026-09-28`.
2. Wait 5–15 minutes. Then check:
   - your email has "Substack draft ready: …" with a link to the draft;
   - Substack shows the draft under **Posts → Drafts**, with a figure above
     each paper's paragraph;
   - the repo has a new commit with
     `vv-paper-fetcher/reports/2026-09-28-substack-post-ste100.md`.

## Every week

Nothing to start. After the Monday digest finishes, the workflow writes the
post and saves it as a draft. You get an email with the link. Open it, review
the post, and click publish in Substack.

## When the cookie expires

You get an email "Substack draft NOT created — <date>" that says the cookie
needs a refresh. The email contains the whole post with the figures attached,
so you can still publish that week by hand.

To fix it:

1. Do step 2 again to copy a new cookie.
2. Update the `SUBSTACK_COOKIE` secret (secrets page → pencil icon → paste →
   **Update secret**).
3. Optional: create that week's draft. Click **Run workflow** and set the
   report date to the date in the email subject.
4. Do step 3 again to set a reminder for the new expiry date.

## If cookie renewal becomes a chore

Two changes can reduce or remove it. Ask Claude to build one:

- **Automatic refresh:** if Substack sends a fresh cookie when the workflow
  uses the old one, the workflow can save it back into the secret. This needs
  a GitHub token that can write repo secrets, and it only helps if Substack
  really does refresh the cookie (not verified yet).
- **Run the Substack step on your Mac:** the script reads the cookie directly
  from Chrome, so you never copy it while you stay signed in. This also avoids
  any Cloudflare block on GitHub's servers. Your Mac must be on and awake on
  Monday morning.

## Optional

- **Style example.** The writer copies the style of one approved post, set by
  `substack_post.style_example` in `config.yaml` (now
  `reports/2026-09-07-substack-post-ste100.md`). When you publish a post you
  like better, point this setting at it.
- **Local runs.** To run `write_post.py` on your Mac without `--dry-run`, add
  `SUBSTACK_PUBLICATION_URL`, `SUBSTACK_COOKIE`, and optionally
  `SUBSTACK_WRITER_MODEL` to `vv-paper-fetcher/.env`. That file is
  gitignored.
