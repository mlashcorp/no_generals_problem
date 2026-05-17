# Blog Operations Runbook

This file is the operational source of truth for writing, previewing, and publishing blog posts in this repository.

## Purpose

- Capture repeatable workflow steps.
- Preserve user preferences for how the blog is operated.
- Reduce ambiguity for future edits and deployments.

## Canonical files

- Writing style: `blog/blog_writing_style.md`
- Agent navigation: `AGENTS.md`
- Site config: `config.yml`
- Production deploy workflow: `.github/workflows/deploy_gh_pages.yaml`
- Preview deploy workflow: `.github/workflows/preview_cloudflare_pages.yaml`

## Current platform setup

- Production host: GitHub Pages
- Preview host: Cloudflare Pages (private via Cloudflare Access)
- Blog title: "The No Generals Problem"
- Homepage byline: "Some thoughts on distributed systems and AI."

## Branching and preview policy

- Use `preview/<topic>` branches for draft preview work.
- Cloudflare preview workflow auto-runs on pushes to `preview/**`.
- Branch alias URL is the stable preview URL for that branch.
- Ignore hash deployment URLs for review bookmarks; use alias URL.

Example stable alias pattern:

- `https://preview-post01-draft.no-generals-problem-preview.pages.dev`

## Post authoring workflow

1. Create or switch to preview branch:

   ```bash
   git checkout -b preview/<topic>
   ```

2. Edit post at `content/posts/<id>/index.md`.
3. Keep draft behavior intentional:
   - For preview-only branch work, `draft: false` is allowed.
   - For production branch (`main`), set `draft: true` until ready to publish.
4. Place post images in `content/posts/<id>/images/`.
   - Image workflow rule: create and iterate images first in `playground/projects/blog/post01_pre_norm/images/`.
   - Move images into `content/posts/<id>/images/` only when explicitly requested.
5. Commit and push preview branch.
6. Wait for Cloudflare preview workflow completion.
7. Review via branch alias URL.

## Publish workflow (production)

1. Confirm content and assets are final.
2. Ensure frontmatter is correct for release (`draft: false`).
3. Merge to `main`.
4. Confirm GitHub Pages deploy succeeds.
5. Confirm `gh-pages` update/publish completes.

## Known gotchas and fixes

### Duplicate title rendering

- Cause: same title in frontmatter and first markdown H1.
- Rule: keep canonical title in frontmatter; avoid repeating same top-level H1 at top.

### Missing image in preview

- Cause: broken path or missing file.
- Rule: verify each referenced image exists under `content/posts/<id>/images/`.

### Cloudflare preview project mismatch

- Symptom: `Project not found` in wrangler logs.
- Fix: verify GitHub variables:
  - `CLOUDFLARE_PAGES_PROJECT_NAME`
  - `CLOUDFLARE_ACCOUNT_ID`

### Different preview URLs every deploy

- Expected behavior: hash URL changes each deployment.
- Use branch alias URL for stable access.

## Preferences to preserve

- Keep blog identity broad (distributed systems + AI intersection).
- Prioritize technical depth over superficial summaries.
- Favor reproducibility and explicit evidence in posts.
- Use private preview before public publish.
- Do not use markdown horizontal rules (`---`) inside post bodies.

## Optional enhancements

- Add a small validation script for title/image checks before preview push.
- Add companion repo links per post in a standard "Reproducibility" section.
