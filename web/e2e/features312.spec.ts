import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test } from "@playwright/test";

// 3.12: thresholds per group in use: Try it decides each row with its group's threshold.
const edges = (ids: string[]) => ids.slice(1).map((b, i) => ({ from: `${ids[i]}/out`, to: `${b}/in` }));

/** east's features are measured low: a model without the region under-scores it. */
function shiftedTable(): string {
  let seed = 1;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const rows = ["a,b,region,label"];
  for (let i = 0; i < 1200; i++) {
    let [a, b] = [rand(), rand()];
    const region = ["north", "south", "east"][i % 3];
    const label = a + b > 1 ? 1 : 0;
    if (region === "east") [a, b] = [a * 0.6, b * 0.6];
    rows.push(`${a.toFixed(3)},${b.toFixed(3)},${region},${label}`);
  }
  const file = join(mkdtempSync(join(tmpdir(), "aime-312-")), "shifted.csv");
  writeFileSync(file, rows.join("\n") + "\n");
  return file;
}

test("thresholds per group, used by Try it", async ({ page, request }) => {
  test.setTimeout(300_000);
  const design = {
    schema_version: 3, name: "e2e_312", meta: {},
    nodes: [
      { id: "in", type: "core.input", params: { shape: "2" }, position: [0, 0] },
      { id: "d1", type: "core.dense", params: { units: 16 }, position: [0, 0] },
      { id: "act", type: "core.relu", params: {}, position: [0, 0] },
      { id: "d2", type: "core.dense", params: { units: 2 }, position: [0, 0] },
      { id: "out", type: "core.output", params: {}, position: [0, 0] },
      { id: "data", type: "data.csv", params: { path: shiftedTable(), target_column: "label" },
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
  const started = await request.post("/api/runs", { data: { graph: design, project: "e2e312" } });
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
  const thresholds = page.getByTestId("thresholds");
  await expect(thresholds).toContainText("A threshold per group", { timeout: 120_000 });
  await thresholds.getByTestId("thresholds-adopt").click();
  await expect(thresholds.getByTestId("thresholds-use")).toContainText("in use, by region");
  await page.keyboard.press("Escape");

  await page.getByTestId("try-run").click();
  const dialog = page.getByRole("dialog", { name: `Try run ${run_id}` });
  await expect(dialog.getByTestId("try-form")).toHaveAttribute("data-kind", "records", { timeout: 60_000 });
  await dialog.locator("#try-a").fill("0.3");
  await dialog.locator("#try-b").fill("0.3");
  await dialog.locator("#try-region").fill("east");
  await expect(dialog.getByTestId("try-group-help")).toContainText("east");
  await dialog.getByTestId("try-predict").click();
  await expect(dialog.getByTestId("try-threshold")).toContainText("east's threshold", { timeout: 60_000 });
  await dialog.getByRole("button", { name: "Close" }).click();

  const off = await request.delete(`/api/runs/${run_id}/group-thresholds`);
  expect((await off.json()).removed).toBe(true);
  await request.delete(`/api/runs/${run_id}`);
});
