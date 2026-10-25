import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test } from "@playwright/test";

import { fileMenu } from "./helpers";

// 3.8: find a learning rate from the optimizer block; Train more and Results by group
// from a finished run.
const edges = (ids: string[]) => ids.slice(1).map((b, i) => ({ from: `${ids[i]}/out`, to: `${b}/in` }));

const tiny = {
  schema_version: 3, name: "e2e_lr_find", meta: {},
  nodes: [
    { id: "in", type: "core.input", params: { shape: "8" }, position: [0, 0] },
    { id: "d1", type: "core.dense", params: { units: 16 }, position: [200, 0] },
    { id: "act", type: "core.relu", params: {}, position: [400, 0] },
    { id: "d2", type: "core.dense", params: { units: 3 }, position: [600, 0] },
    { id: "out", type: "core.output", params: {}, position: [800, 0] },
    { id: "data", type: "data.synthetic", position: [0, 200],
      params: { kind: "classification", n_samples: 400, n_features: 8, n_classes: 3, seed: 3 } },
    { id: "opt", type: "train.adam", params: { lr: 0.5 }, position: [200, 200] },
    { id: "loss", type: "train.loss_cross_entropy", params: {}, position: [400, 200] },
    { id: "trainer", type: "train.trainer", params: { epochs: 2, batch_size: 16, device: "cpu" },
      position: [600, 200] },
  ],
  edges: edges(["in", "d1", "act", "d2", "out"]),
};

function regionTable(): string {
  let seed = 1;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const rows = ["a,b,region,label"];
  for (let i = 0; i < 600; i++) {
    const [a, b] = [rand(), rand()];
    const region = ["north", "south", "east"][i % 3];
    let label = a + b > 1 ? 1 : 0;
    if (region === "east" && rand() < 0.45) label = 1 - label;   // noisy in one region
    rows.push(`${a.toFixed(3)},${b.toFixed(3)},${region},${label}`);
  }
  const file = join(mkdtempSync(join(tmpdir(), "aime-groups-")), "regions.csv");
  writeFileSync(file, rows.join("\n") + "\n");
  return file;
}

test("find a learning rate from the optimizer block and use it", async ({ page, request }) => {
  test.setTimeout(180_000);
  const put = await request.put("/api/projects/e2e_lr_find", { data: tiny });
  expect(put.ok(), await put.text()).toBeTruthy();
  try {
    await page.goto("/");
    await expect(page.locator(".block-node").first()).toBeVisible();
    await fileMenu(page, "open-project");
    await page.locator(".issue", { hasText: "e2e_lr_find" }).click();
    await expect(page.getByTestId("status")).toHaveText("No problems", { timeout: 30_000 });
    await page.locator(".block-node", { hasText: "Adam" }).first().click();
    await page.getByTestId("find-lr").click();
    const summary = page.getByTestId("lr-summary");
    await expect(summary).toContainText("Suggested", { timeout: 150_000 });
    await expect(page.getByTestId("chart-marker")).toBeVisible();
    const label = (await page.getByTestId("use-lr").textContent()) ?? "";
    await page.getByTestId("use-lr").click();
    const value = Number(await page.locator("#param-lr").inputValue());
    expect(value).toBeGreaterThan(1e-7);
    expect(value).toBeLessThan(10);
    expect(value).not.toBe(0.5);
    expect(label).toContain("Use");
  } finally {
    await request.delete("/api/projects/e2e_lr_find");
  }
});

test("results by group flag the weak region, and a run trains more", async ({ page, request }) => {
  test.setTimeout(240_000);
  const design = {
    schema_version: 3, name: "e2e_regions", meta: {},
    nodes: [
      { id: "in", type: "core.input", params: { shape: "2" }, position: [0, 0] },
      { id: "d1", type: "core.dense", params: { units: 16 }, position: [0, 0] },
      { id: "act", type: "core.relu", params: {}, position: [0, 0] },
      { id: "d2", type: "core.dense", params: { units: 2 }, position: [0, 0] },
      { id: "out", type: "core.output", params: {}, position: [0, 0] },
      { id: "data", type: "data.csv", params: { path: regionTable(), target_column: "label" },
        position: [0, 0] },
      { id: "drop", type: "prep.drop_columns", params: { columns: "region" }, position: [0, 0] },
      { id: "opt", type: "train.adam", params: { lr: 0.02 }, position: [0, 0] },
      { id: "loss", type: "train.loss_cross_entropy", params: {}, position: [0, 0] },
      { id: "acc", type: "eval.accuracy", params: {}, position: [0, 0] },
      { id: "trainer", type: "train.trainer",
        params: { epochs: 8, batch_size: 32, device: "cpu", seed: 0 }, position: [0, 0] },
    ],
    edges: edges(["in", "d1", "act", "d2", "out"]),
  };
  const started = await request.post("/api/runs", { data: { graph: design, project: "e2e38" } });
  expect(started.ok(), await started.text()).toBeTruthy();
  const { run_id } = await started.json();
  await expect.poll(async () => (await (await request.get(`/api/runs/${run_id}`)).json()).status,
    { timeout: 180_000 }).toBe("finished");

  await page.goto("/");
  await page.getByRole("button", { name: "Experiments" }).click();
  await page.getByLabel(/only “/).uncheck();          // runs of every project
  await page.getByTestId("runs").locator("tr", { hasText: run_id }).click();
  await page.getByTestId("groups-run").click();
  await page.locator("#group-column").selectOption("region");
  await page.getByTestId("score-groups").click();
  const table = page.getByTestId("group-results");
  await expect(table).toContainText("Worse for east", { timeout: 120_000 });
  await expect(table.locator("tr.flagged")).toHaveCount(1);
  await expect(page.getByTestId("fairness")).toContainText(/Treated differently: .*east/); // 3.10
  await page.keyboard.press("Escape");

  await page.getByTestId("train-more").click();
  await page.locator("#more-epochs").fill("2");
  await page.getByTestId("start-more").click();
  await expect.poll(async () => {
    const runs = (await (await request.get("/api/runs")).json()).runs as
      { note: string; tags: string[]; status: string }[];
    return runs.find((r) => r.tags.includes("train-more") && r.note.includes(run_id))?.status;
  }, { timeout: 120_000 }).toBe("finished");
});
