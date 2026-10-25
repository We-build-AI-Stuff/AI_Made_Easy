import { expect, test } from "@playwright/test";
import { fileURLToPath } from "node:url";

import { fileMenu } from "./helpers";

// Docs screenshots of the 3.2 web tools: AIME_SCREENSHOTS=1 npx playwright test screenshots
const OUT = fileURLToPath(new URL("../../docs/images", import.meta.url));
test.skip(!process.env.AIME_SCREENSHOTS, "set AIME_SCREENSHOTS=1 to refresh the docs images");

const tiny = {
  name: "churn_model",
  meta: { budget: { device: "raspberry_pi_5" } },
  nodes: [
    { id: "in", type: "core.input", params: { shape: "8" }, position: [0, 0] },
    { id: "d1", type: "core.dense", params: { units: 16 }, position: [0, 0] },
    { id: "act", type: "core.relu", params: {}, position: [0, 0] },
    { id: "d2", type: "core.dense", params: { units: 3 }, position: [0, 0] },
    { id: "out", type: "core.output", params: {}, position: [0, 0] },
    { id: "data", type: "data.synthetic", position: [0, 0],
      params: { kind: "classification", n_samples: 200, n_features: 8, n_classes: 3, seed: 3 } },
    { id: "opt", type: "train.adam", params: { lr: 0.01 }, position: [0, 0] },
    { id: "loss", type: "train.loss_cross_entropy", params: {}, position: [0, 0] },
    { id: "trainer", type: "train.trainer", params: { epochs: 3, batch_size: 32, device: "cpu" },
      position: [0, 0] },
  ],
  edges: [["in", "d1"], ["d1", "act"], ["act", "d2"], ["d2", "out"]]
    .map(([a, b]) => ({ from: `${a}/out`, to: `${b}/in` })),
};

test("docs screenshots: Tools and Share", async ({ page, request }) => {
  const started = await (await request.post("/api/runs", { data: { graph: tiny, project: "churn" } })).json();
  await expect.poll(async () => (await (await request.get(`/api/runs/${started.run_id}`)).json()).status,
    { timeout: 120_000 }).toBe("finished");
  const model = await (await request.post("/api/models", { data: { run_id: started.run_id, name: "churn-model" } })).json();

  await page.goto("/");
  await expect(page.locator(".block-node").first()).toBeVisible();
  await page.getByTestId("open-tools").click();
  await page.getByTestId("dataset-source").fill("kaggle:blastchar/telco-customer-churn");
  await page.getByRole("dialog").screenshot({ path: `${OUT}/web_tools.png` });
  await page.getByRole("dialog").getByRole("button", { name: "Close" }).click();

  await page.getByRole("button", { name: "Models", exact: true }).click();
  await page.getByTestId(`share-churn-model-${model.version}`).click();
  const share = page.getByRole("dialog", { name: /Share churn-model/ });
  await expect(share.locator("[data-format]").first()).toBeVisible();
  await share.getByTestId("share-repo").fill("your-name/churn-model");
  await share.screenshot({ path: `${OUT}/web_share.png` });
  await share.getByRole("button", { name: "Close" }).click();
  await request.delete(`/api/models/churn-model/${model.version}`);
});

test("docs screenshots: 3.3 command palette and canvas notes / groups", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => localStorage.removeItem("aime.autosave"));
  await page.reload();
  const blocks = page.locator(".block-node");
  await expect(blocks.first()).toBeVisible();
  await blocks.nth(1).click();
  await blocks.nth(2).click({ modifiers: ["ControlOrMeta"] });
  await blocks.nth(3).click({ modifiers: ["ControlOrMeta"] });
  await page.getByTestId("group").click();
  await page.getByTestId("group-group1").getByRole("button", { name: "Rename Group" }).click();
  await page.getByLabel("Group name").fill("Feature extractor");
  await page.getByLabel("Group name").press("Enter");
  await page.getByTestId("add-note").click();
  await page.getByTestId("note-text-note1").fill("Try 32 filters in the first conv");
  await page.locator(".react-flow__pane").click({ position: { x: 5, y: 5 } });
  await page.keyboard.press("ControlOrMeta+f");      // centre on the group at 100 %
  await page.getByTestId("palette").getByRole("combobox").fill("@relu");
  await page.keyboard.press("Enter");
  await page.waitForTimeout(400);
  await page.locator(".react-flow__pane").click({ position: { x: 5, y: 5 } });
  await page.locator(".canvas").screenshot({ path: `${OUT}/web_canvas_groups.png` });
  await page.keyboard.press("ControlOrMeta+k");
  await page.getByTestId("palette").getByRole("combobox").fill("export");
  await page.screenshot({ path: `${OUT}/web_palette.png` });
});

test("docs screenshots: 3.5 Try it", async ({ page, request }) => {
  const started = await (await request.post("/api/runs", { data: { graph: tiny, project: "churn" } })).json();
  const runId: string = started.run_id;
  await expect.poll(async () => (await (await request.get(`/api/runs/${runId}`)).json()).status,
    { timeout: 120_000 }).toBe("finished");
  await page.goto("/");
  await expect(page.locator(".block-node").first()).toBeVisible();
  await page.getByRole("button", { name: "Experiments" }).click();
  await page.getByLabel(/only/).uncheck();
  await page.getByTestId("runs").locator("tr", { hasText: runId }).click();
  await page.getByTestId("try-run").click();
  const dialog = page.getByRole("dialog", { name: `Try run ${runId}` });
  await expect(dialog.getByTestId("try-form")).toBeVisible({ timeout: 60_000 });
  await dialog.getByTestId("try-json").fill(JSON.stringify([[0.4, -1.2, 0.8, 0.1, 1.5, -0.3, 0.9, -0.7]]));
  await dialog.getByTestId("try-predict").click();
  await expect(dialog.getByTestId("try-result")).toBeVisible({ timeout: 60_000 });
  await dialog.screenshot({ path: `${OUT}/web_try_it.png` });
  await request.delete(`/api/runs/${runId}`);
});

test("docs screenshots: 3.6 which settings mattered", async ({ page, request }) => {
  test.setTimeout(300_000);
  const spec = {
    metric: "val_loss", strategy: "random", max_trials: 8, seed: 1,
    dimensions: [
      { node: "opt", param: "lr", kind: "float", low: 0.0003, high: 0.3, log: true },
      { node: "d1", param: "units", kind: "choice", values: [4, 16, 64] },
      { node: "trainer", param: "batch_size", kind: "choice", values: [16, 32] }],
  };
  const started = await (await request.post("/api/sweeps",
    { data: { graph: { ...tiny, name: "churn_lr_sweep" }, spec, project: "churn" } })).json();
  await expect.poll(async () => (await (await request.get(`/api/sweeps/${started.sweep_id}`)).json()).state,
    { timeout: 280_000 }).toBe("finished");
  await page.goto("/");
  await expect(page.locator(".block-node").first()).toBeVisible();
  await page.getByRole("button", { name: "Experiments" }).click();
  await page.getByRole("button", { name: "Sweeps" }).click();
  await page.getByText(/churn_lr_sweep/).first().click();
  const insights = page.getByTestId("sweep-insights");
  await expect(insights.getByTestId("parallel-chart").locator("polyline")).toHaveCount(8);
  await insights.screenshot({ path: `${OUT}/web_sweep_insights.png` });
});

test("docs screenshots: 3.7 which settings matter together", async ({ page, request }) => {
  test.setTimeout(480_000);
  const spec = {
    metric: "val_loss", strategy: "random", max_trials: 12, seed: 4,
    dimensions: [
      { node: "opt", param: "lr", kind: "float", low: 0.0003, high: 0.3, log: true },
      { node: "d1", param: "units", kind: "choice", values: [4, 16, 64] },
      { node: "trainer", param: "batch_size", kind: "choice", values: [16, 64] }],
  };
  const started = await (await request.post("/api/sweeps",
    { data: { graph: { ...tiny, name: "churn_pairs_sweep" }, spec, project: "churn" } })).json();
  await expect.poll(async () => (await (await request.get(`/api/sweeps/${started.sweep_id}`)).json()).state,
    { timeout: 460_000 }).toBe("finished");
  await page.goto("/");
  await expect(page.locator(".block-node").first()).toBeVisible();
  await page.getByRole("button", { name: "Experiments" }).click();
  await page.getByRole("button", { name: "Sweeps" }).click();
  await page.getByText(/churn_pairs_sweep/).first().click();
  const insights = page.getByTestId("sweep-insights");
  await expect(insights.getByTestId("pair-heatmap")).toBeVisible({ timeout: 30_000 });   // the first analysis imports scikit-learn
  await insights.screenshot({ path: `${OUT}/web_sweep_pairs.png` });
});

test("docs screenshots: 3.8 find a learning rate", async ({ page, request }) => {
  test.setTimeout(240_000);
  const at = (i: number, row = 0): [number, number] => [i * 200, row * 200];
  const nodes = tiny.nodes.map((n, i) => ({ ...n, position: i < 5 ? at(i) : at(i - 5, 1) }));
  const graph = { ...tiny, schema_version: 3, name: "churn_lr_find", nodes };
  const put = await request.put("/api/projects/churn_lr_find", { data: graph });
  expect(put.ok(), await put.text()).toBeTruthy();
  try {
    await page.goto("/");
    await expect(page.locator(".block-node").first()).toBeVisible();
    await fileMenu(page, "open-project");
    await page.locator(".issue", { hasText: "churn_lr_find" }).click();
    await expect(page.getByTestId("status")).toHaveText("No problems", { timeout: 30_000 });
    await page.locator(".block-node", { hasText: "Adam" }).first().click();
    await page.getByTestId("find-lr").click();
    await expect(page.getByTestId("lr-summary")).toContainText("Suggested", { timeout: 200_000 });
    await page.getByRole("dialog").screenshot({ path: `${OUT}/web_lr_finder.png` });
  } finally {
    await request.delete("/api/projects/churn_lr_find");
  }
});

test("docs screenshots: 3.9 train a boosted model more", async ({ page, request }) => {
  test.setTimeout(240_000);
  const { mkdtempSync, writeFileSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  let seed = 1;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const rows = ["tenure,spend,plan,churned"];
  for (let i = 0; i < 400; i++) {
    const [t, s] = [rand(), rand()];
    rows.push(`${(t * 60).toFixed(0)},${(s * 120).toFixed(1)},${["basic", "plus", "pro"][i % 3]},${t + s < 0.9 ? 1 : 0}`);
  }
  const table = join(mkdtempSync(join(tmpdir(), "aime-shot-")), "churn.csv");
  writeFileSync(table, rows.join("\n") + "\n");
  const design = {
    schema_version: 3, name: "churn_xgboost", meta: {},
    nodes: [
      { id: "data", type: "data.csv", params: { path: table, target_column: "churned" }, position: [0, 0] },
      { id: "encode", type: "prep.one_hot", params: {}, position: [0, 0] },
      { id: "model", type: "ml.xgb_classifier", params: { n_estimators: 100 }, position: [0, 0] },
      { id: "acc", type: "eval.accuracy", params: {}, position: [0, 0] },
    ],
    edges: [],
  };
  const started = await (await request.post("/api/runs", { data: { graph: design, project: "churn" } })).json();
  await expect.poll(async () => (await (await request.get(`/api/runs/${started.run_id}`)).json()).status,
    { timeout: 200_000 }).toBe("finished");
  await page.goto("/");
  await page.getByRole("button", { name: "Experiments" }).click();
  await page.getByLabel(/only “/).uncheck();
  await page.getByTestId("runs").locator("tr", { hasText: started.run_id }).click();
  await page.getByTestId("train-more").click();
  await page.getByRole("dialog").screenshot({ path: `${OUT}/web_train_more_trees.png` });
});

test("docs screenshots: 3.10 fairness by group", async ({ page, request }) => {
  test.setTimeout(300_000);
  const { mkdtempSync, writeFileSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const rows = ["tenure,spend,region,churned"];
  for (let i = 0; i < 900; i++) {
    const [t, s] = [rand(), rand()];
    const region = ["north", "south", "east"][i % 3];
    let label = t + s < 1 ? 1 : 0;
    if (region === "east" && rand() < 0.4) label = 1 - label;   // the model can't learn east
    rows.push(`${(t * 60).toFixed(0)},${(s * 120).toFixed(1)},${region},${label}`);
  }
  const table = join(mkdtempSync(join(tmpdir(), "aime-shot-")), "churn_regions.csv");
  writeFileSync(table, rows.join("\n") + "\n");
  const chain = ["in", "d1", "act", "d2", "out"];
  const design = {
    schema_version: 3, name: "churn_regions", meta: {},
    nodes: [
      { id: "in", type: "core.input", params: { shape: "2" }, position: [0, 0] },
      { id: "d1", type: "core.dense", params: { units: 16 }, position: [0, 0] },
      { id: "act", type: "core.relu", params: {}, position: [0, 0] },
      { id: "d2", type: "core.dense", params: { units: 2 }, position: [0, 0] },
      { id: "out", type: "core.output", params: {}, position: [0, 0] },
      { id: "data", type: "data.csv", params: { path: table, target_column: "churned" }, position: [0, 0] },
      { id: "drop", type: "prep.drop_columns", params: { columns: "region" }, position: [0, 0] },
      { id: "opt", type: "train.adam", params: { lr: 0.02 }, position: [0, 0] },
      { id: "loss", type: "train.loss_cross_entropy", params: {}, position: [0, 0] },
      { id: "acc", type: "eval.accuracy", params: {}, position: [0, 0] },
      { id: "trainer", type: "train.trainer",
        params: { epochs: 8, batch_size: 32, device: "cpu", seed: 0 }, position: [0, 0] },
    ],
    edges: chain.slice(1).map((b, i) => ({ from: `${chain[i]}/out`, to: `${b}/in` })),
  };
  const started = await (await request.post("/api/runs", { data: { graph: design, project: "churn" } })).json();
  await expect.poll(async () => (await (await request.get(`/api/runs/${started.run_id}`)).json()).status,
    { timeout: 200_000 }).toBe("finished");
  await page.goto("/");
  await page.getByRole("button", { name: "Experiments" }).click();
  await page.getByLabel(/only “/).uncheck();
  await page.getByTestId("runs").locator("tr", { hasText: started.run_id }).click();
  await page.getByTestId("groups-run").click();
  await page.locator("#group-column").selectOption("region");
  await page.getByTestId("score-groups").click();
  await expect(page.getByTestId("fairness")).toBeVisible({ timeout: 120_000 });
  await page.getByRole("dialog").screenshot({ path: `${OUT}/web_fairness_by_group.png` });
});
