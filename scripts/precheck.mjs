/* -------------------------------------------------------------------------
 * The checks that run on every pull request (.github/workflows/precheck.yml)
 * and on your machine before a push:
 *
 *   1. the Astro type check (`npm run typecheck`) and a production build
 *   2. every URL in the sitemap answers 200 on the built site
 *   3. the six test devices (scripts/precheck-render.py): no horizontal
 *      overflow, no console errors, exactly one h1, no broken image, an alt on
 *      every image, and no cover or illustration cut off by its box
 *
 *     npm run precheck
 *
 * Empty output is a failure, not a pass: every step prints what it counted.
 * Needs uv and Chromium for step 3: set PLAYWRIGHT_CHROME, or run
 * `uvx --from playwright==1.55.0 playwright install chromium`.
 * ------------------------------------------------------------------------- */
import { spawn, spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";

const PORT = 4399;
const BASE = `http://127.0.0.1:${PORT}`;
const failures = [];
const fail = (message) => {
  failures.push(message);
  console.error(`  FAIL ${message}`);
};

function run(label, command, args) {
  console.log(`\n== ${label}`);
  const result = spawnSync(command, args, { stdio: "inherit" });
  if (result.status !== 0) {
    console.error(`precheck: ${label} failed`);
    process.exit(1);
  }
}

run("type check", "npm", ["run", "typecheck"]);
run("build", "npm", ["run", "build"]);

console.log("\n== server");
const server = spawn("node_modules/.bin/astro", ["preview", "--host", "127.0.0.1", "--port", String(PORT)], {
  stdio: "ignore",
  detached: true,
});
const stop = () => {
  try {
    process.kill(-server.pid);
  } catch {}
};
process.on("exit", stop);
process.on("SIGINT", () => process.exit(130));

let up = false;
for (let i = 0; i < 60 && !up; i++) {
  await new Promise((resolve) => setTimeout(resolve, 500));
  up = await fetch(BASE).then((r) => r.ok).catch(() => false);
}
if (!up) {
  console.error("precheck: the server did not start");
  process.exit(1);
}
console.log(`  up on ${BASE}`);

console.log("\n== sitemap");
const sitemap = readFileSync("dist/sitemap-0.xml", "utf8");
const paths = [...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map((m) => new URL(m[1]).pathname || "/");
if (paths.length < 50) fail(`the sitemap lists only ${paths.length} URLs`);
for (const path of paths) {
  const res = await fetch(BASE + path, { redirect: "manual" });
  if (res.status !== 200) fail(`${path} answered ${res.status}, want 200`);
}
console.log(`  ${paths.length} sitemap URLs checked`);

console.log("\n== devices, light and dark");
const render = spawnSync(
  "uv",
  ["run", "--no-project", "--with", "playwright==1.55.0", "python", "scripts/precheck-render.py", BASE],
  { stdio: "inherit" },
);
if (render.status !== 0) fail("the rendered check failed");

if (failures.length) {
  console.error(`\nprecheck: ${failures.length} problem(s)`);
  process.exit(1);
}
console.log("\nprecheck: ok. Say in the PR which steps ran.");
