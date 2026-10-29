# AGENTS.md: taewoon.kim

Taewoon Kim's personal website (Astro 5, static, published to GitHub Pages by `.github/workflows/deploy.yml` on every push to `main`). Start with `README.md`.

## Rules for agents

- Before a push that touches posts, pages, styles or scripts, run `npm run precheck` (a few minutes: type check, build, every sitemap URL, every page on six devices) and say in the PR which steps ran. The same check runs on every such pull request in GitHub Actions and blocks the merge when it is red.
- A post is `src/content/blog/YYYY-MM-DD-slug.md`; the URL is `/YYYY/MM/DD/slug/`. `cover-img` is the image shown whole on the post page. `thumbnail-img` is the card image: when the cover is not 2:1, make a 2:1 thumbnail that shows it whole (the card crops to 2:1, and `npm run precheck` fails on a cover or illustration that the crop cuts). Refit an image onto a canvas (crop to the content, scale, recenter); do not redraw or recolor it.
- Cards on the front page are image, date, title, subtitle, a short excerpt and a link, the same size in a row. Tags belong on the post page and the tag pages, not on the cards.
- A post about a paper links the paper and the research page on humem.ai, and nothing more: no BibTeX, no DOI, no list of every repository. Say "research page" or "software page" for humem.ai pages, never "project page".
- Write plain, first-person prose. No em dashes, no template headings ("Why this matters"), no closing recap. A post that announces something says it in the title ("My paper was accepted at X").
- Python is run with `uv` (`uv run --no-project --with <pkg> ...`), never `pip` or a bare `python3`.
- Open an issue, work on a branch, open a PR with `Closes #N`, merge when the checks are green, delete the branch on merge. Stage files by explicit path. Issue, PR and comment bodies have no hard wraps and no em dashes.
- This repository is public: no secrets, token locations or private notes.
