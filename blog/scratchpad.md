# Blog Scratchpad

## Repository overview

- Stack: Hugo static site with the PaperMod theme.
- Main config: `config.yml`.
- Post content path: `content/posts/<post-id>/index.md`.
- Post assets path: `content/posts/<post-id>/` (images live next to `index.md`).
- Deploy workflow: `.github/workflows/deploy_gh_pages.yaml`.
- GitHub Pages source: `gh-pages` branch.

## Current post status and URL

- Post source file: `content/posts/1/index.md`
- Current status: `draft: true` (not publicly visible in production builds).
- Published URL when `draft: false`:
  - `https://mlashcorp.github.io/no_generals_problem/posts/1/`

## How to create a new blog post

1. Create a new post folder and markdown file:

   ```bash
   mkdir -p content/posts/2
   cp content/posts/1/index.md content/posts/2/index.md
   ```

2. Edit frontmatter in `content/posts/2/index.md`:
   - Update `title`
   - Update `date`
   - Keep `draft: true` while writing

3. Add images for the post into `content/posts/2/` and reference by filename:

   ```md
   ![diagram](diagram.png)
   ```

4. Preview locally (including drafts):

   ```bash
   hugo server -D
   ```

5. Publish the post:
   - Set `draft: false`
   - Commit and push to `main`
   - GitHub Action builds `public/` and updates `gh-pages`

## Deployment notes

- Pushes to `main` trigger deployment.
- The workflow builds with:

  ```bash
  hugo --minify
  ```

- The deploy step publishes `./public` to `gh-pages` using `peaceiris/actions-gh-pages`.

## Private preview hosting (Cloudflare Pages)

- Workflow: `.github/workflows/preview_cloudflare_pages.yaml`
- Trigger: pull requests and manual run (`workflow_dispatch`).
- Result: deploys a preview build to Cloudflare Pages and comments the preview URL on the PR.

### Required GitHub settings

1. Repository secret `CLOUDFLARE_API_TOKEN`
   - Create in Cloudflare with `Cloudflare Pages:Edit` permission for the target account.
2. Repository variable `CLOUDFLARE_PAGES_PROJECT_NAME`
   - Example: `no-generals-problem-preview`

### Required Cloudflare Access setup (to keep previews private)

1. In Cloudflare Zero Trust, create an Access application for `*.pages.dev` host of your project.
2. Add policy allowing only your email (or your team emails).
3. Keep production GitHub Pages public; this affects only preview URLs on Cloudflare.

## Useful commands

```bash
hugo server -D
hugo --minify
git status
git add -A
git commit -m "<message>"
git push origin main
```
