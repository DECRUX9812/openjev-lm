# open-Jev — go-live checklist

**Status 2026-09-18:** step 1 and 3 are **DONE** — the repos are public and Pages is live:
`https://github.com/DECRUX9812/openjev-lm` · `https://decrux9812.github.io/openjev-lm/`.
Remaining: HF upload (needs the token file), the X thread, HN/Reddit (those are your accounts).

## 1. Publish the code — DONE

- [x] LM arm pushed → https://github.com/DECRUX9812/openjev-lm (default branch `main`)
- [x] classifier arm pushed → ledger correction `424198d` on `DECRUX9812/openjev`

## 3. Pages — DONE

- [x] `docs/` published, build status `built`, landing page + paper + figures all return 200.

## 0. Preconditions (once)

- [ ] `gh` is authenticated as **DECRUX9812** (`gh auth status`) — used for the repo + Pages.
- [ ] Hugging Face token created (`https://huggingface.co/settings/tokens`, *write* scope) and
      saved to `~/.hermes/.hf_token` with `chmod 600` (file only — never paste it anywhere else).
- [ ] Decide who posts the X thread: **you** (recommended — paste from `launch/x_thread.md`,
      upload the three images) or an API client if one is already configured.

## 2. Hugging Face (3 commands)

```bash
cd ~/Code/openjev-lm
export HF_TOKEN="$(cat ~/.hermes/.hf_token)"          # shell only; never committed
python hf/upload_model.py   --repo-id DECRUX9812/openjev-0.5b          --dry-run   # verify plan
python hf/upload_model.py   --repo-id DECRUX9812/openjev-0.5b          --go
python hf/upload_dataset.py --repo-id DECRUX9812/openjev-jev-labelled  --dry-run   # verify plan
python hf/upload_dataset.py --repo-id DECRUX9812/openjev-jev-labelled  --go
```

Both dry runs are already verified locally: the model plan stages 3/3 files and the adapter's
sha256 matches the pinned source hash; the dataset plan stages 4/4 files (2,591 + 286 + 70 rows).

Verify the model page renders, the card frontmatter parses, and the adapter downloads.

## 3. Landing page (GitHub Pages)

```bash
gh api -X POST repos/DECRUX9812/openjev-lm/pages \
  -f 'source[branch]=main' -f 'source[path]=/docs'
# then: https://decrux9812.github.io/openjev-lm/
```

- [ ] Page loads, both figures render, the paper PDF link works.
- [ ] Add the Pages URL to the top of the README, commit, push.

## 4. The thread (the actual launch)

Images: `paper/xcards/card1.png` → tweet 1 · `paper/figs/fig1_ladder.png` → tweet 2 ·
`paper/xcards/card2.png` → tweet 6.

- [ ] Replace `{PAPER}` `{REPO}` `{HF}` in `launch/x_thread.md` with the live URLs
      (Pages URL for the paper, repo, HF model).
- [ ] Post tweets 1–14 in order, images attached as mapped.
- [ ] Pin tweet 1.

## 5. Distribution (same hour, while the thread is hot)

- [ ] **Show HN** — paste `launch/showhn.md` (link to the repo; mention the paper in the body).
      Best window: weekday morning US-Eastern.
- [ ] **r/LocalLLaMA** — paste `launch/reddit_local_llama.md`.
- [ ] **HF community post** — paste `launch/hf_post.md` on the model page.
- [ ] **LinkedIn** (optional) — paste `launch/linkedin.md`.

## 6. After

- [ ] Reply to every substantive question in the first 6 hours; the receipts in the repo do the
      heavy lifting. Do not dunk on the viral author — link the teardown (`docs/viral-teardown.md`)
      and move on.
- [ ] Add the HN/Reddit links to the README as they land.
- [ ] **arXiv (optional, needs LaTeX):** arXiv does not accept PDF-only submissions for cs.*, so
      this needs a LaTeX pass (paper source is `paper/openjev-paper.md` → ~half a day of
      conversion). Until then the canonical paper is the PDF in the repo + Pages.

## Fail-safe

If anything reads wrong after publishing, the repo is the single place to fix it: correct the
file, note the correction in the commit message, and reply in-thread with the corrected number.
Every number in the paper traces to `runs/FACTS.json` and `launch/check_numbers.py` re-checks
that — run it before any further edits.
