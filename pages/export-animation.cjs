"use strict";

// Export the existing desktop demo without changing the website or its timeline.
// Requires Node.js, Playwright (with Chromium), and ffmpeg.
// Start a server for pages/, then run:
//   node pages/export-animation.cjs http://localhost:8765/
// Optional: CHROMIUM_EXECUTABLE, GIF_FPS, and GIF_COLORS environment variables.
const { chromium } = require("playwright");
const { mkdtemp, mkdir, rm, stat } = require("node:fs/promises");
const { tmpdir } = require("node:os");
const path = require("node:path");
const { spawnSync } = require("node:child_process");

async function main() {
  const url = process.argv[2] || "http://localhost:8765/";
  const output = path.resolve(__dirname, "../assets/steering-demo.gif");
  const fps = Number(process.env.GIF_FPS || 12);
  const colors = Number(process.env.GIF_COLORS || 96);
  const frames = await mkdtemp(path.join(tmpdir(), "steering-gif-"));
  let browser;
  try {
    browser = await chromium.launch({
      headless: true,
      ...(process.env.CHROMIUM_EXECUTABLE ? { executablePath: process.env.CHROMIUM_EXECUTABLE } : {}),
      args: ["--no-sandbox"],
    });
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1100 },
      deviceScaleFactor: 1,
      colorScheme: "light",
      reducedMotion: "no-preference",
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.clock.install();
    // Add a capture-only seek hook in the browser, never in the shipped script.
    await page.route("**/steering-demo.js", async (route) => {
      const response = await route.fetch();
      const source = await response.text();
      const marker = "box.steeringDemo = {";
      if (!source.includes(marker)) throw new Error("Demo export hook no longer matches the source.");
      await route.fulfill({
        response,
        body: source.replace(marker, marker + "\nseekForExport(time) { stopFrame(); elapsed = time; render(); },\ndurationForExport() { return timing.end; },"),
      });
    });
    await page.goto(url, { waitUntil: "load" });
    await page.evaluate(() => document.fonts.ready);
    const demo = page.locator(".steering-demo");
    await demo.scrollIntoViewIfNeeded();
    // Controls are for the live site, not part of the exported illustration.
    await page.locator(".steering-demo-controls").evaluate((node) => { node.style.display = "none"; });
    await page.clock.pauseAt(new Date());
    const end = await demo.evaluate((node) => node.steeringDemo.durationForExport());
    const count = Math.floor(end * fps / 1000);
    for (let index = 0; index < count; index++) {
      const time = index * 1000 / fps;
      await demo.evaluate((node, value) => {
        node.steeringDemo.seekForExport(value);
        for (const animation of node.getAnimations({ subtree: true })) {
          animation.pause();
          animation.currentTime = value;
        }
      }, time);
      await demo.screenshot({
        path: path.join(frames, "frame-" + String(index).padStart(4, "0") + ".png"),
        animations: "allow",
      });
      if (index % fps === 0) console.log("Captured " + Math.round(time / 1000) + "s / " + Math.round(end / 1000) + "s");
    }
    if (errors.length) throw new Error(errors.join("\n"));
    await mkdir(path.dirname(output), { recursive: true });
    const result = spawnSync("ffmpeg", [
      "-y", "-loglevel", "error", "-framerate", String(fps),
      "-i", path.join(frames, "frame-%04d.png"),
      "-filter_complex", "[0:v]split[a][b];[a]palettegen=max_colors=" + colors + ":stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
      "-loop", "0", "-gifflags", "+transdiff", output,
    ], { encoding: "utf8" });
    if (result.status !== 0) throw new Error(result.stderr || "ffmpeg export failed.");
    console.log(output + " — " + Math.round((await stat(output)).size / 1024) + " KiB");
  } finally {
    if (browser) await browser.close();
    await rm(frames, { recursive: true, force: true });
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
