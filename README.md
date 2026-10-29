# taewoon.kim

Personal website built with Astro.

## Development

```bash
npm ci
npm run dev
```

## Checks

```bash
npm run precheck
```

Runs the Astro type check (`npm run typecheck`), the production build, every URL in the sitemap, and the built site on six devices (360, 390, 768, 984, 1280 and 1920 px wide): no horizontal overflow, no console error, exactly one h1, no broken image, an alt on every image, and no cover or illustration cut off by the box it is shown in. It takes a few minutes and needs `uv` and a Chromium (`uvx --from playwright==1.55.0 playwright install chromium`, or set `PLAYWRIGHT_CHROME`).

The same check runs on every pull request that touches the site (`.github/workflows/precheck.yml`), so a red check blocks the merge. The deploy workflow runs only after a push to `main`.

## Build

```bash
npm run build
```

The production build is emitted to `dist/`.

## Deployment

GitHub Pages is configured through GitHub Actions. The workflow lives in `.github/workflows/deploy.yml` and publishes the built `dist/` output.

Keep the repository Pages source set to `GitHub Actions`.
