# AGENTS.md

This file guides coding agents working on blog content in this repository.

## Primary objective

When asked to work on a blog post, prioritize content quality, reproducibility, and safe publish workflow.

## Read these first

1. `blog/blog_writing_style.md`
   - Voice, structure, evidence standards, reproducibility checklist.
2. `blog/TONAL-GUIDANCE.md`
   - Sentence-level tone preferences and concrete before/after examples.
3. `blog/OPERATIONS.md`
   - Branching, preview, publish runbook, and user operation preferences.
4. `blog/scratchpad.md`
   - Repo-specific workflow, deploy notes, preview setup.
5. `content/posts/<post-id>/index.md`
   - The actual post draft to edit.

## Blog content locations

- Post markdown: `content/posts/<post-id>/index.md`
- Post assets: `content/posts/<post-id>/`
- Blog process docs: `blog/`

## Deployment and preview files

- Production deploy (GitHub Pages): `.github/workflows/deploy_gh_pages.yaml`
- Private preview deploy (Cloudflare Pages): `.github/workflows/preview_cloudflare_pages.yaml`
- Site config: `config.yml`

## Working rules for blog edits

- Keep posts in `draft: true` until explicitly ready.
- Do not weaken reproducibility details (commits, configs, commands, seeds).
- Prefer small, auditable edits; preserve technical correctness over style tweaks.
- Keep claims tied to evidence (figures, metrics, logs, code refs).

## Quick workflow

1. Read style + scratchpad docs.
2. Edit post draft and nearby assets only.
3. Validate links/paths and markdown structure.
4. Use Cloudflare preview workflow for private review.
5. Publish only when draft status is explicitly changed.
