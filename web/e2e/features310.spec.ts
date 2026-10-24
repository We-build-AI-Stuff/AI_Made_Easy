import { expect, test } from "@playwright/test";

import { fileMenu } from "./helpers";

// 3.10: a network can train with Keras from the app: the choice is saved on the design, and
// the learning-rate finder follows it.
const edges = (ids: string[]) => ids.slice(1).map((b, i) => ({ from: `${ids[i]}/out`, to: `${b}/in` }));

const tiny = {
  schema_version: 3, name: "e2e_framework", meta: {},
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

test("a design that chooses Keras finds its learning rate with Keras", async ({ page, request }) => {
  test.setTimeout(240_000);
  const put = await request.put("/api/projects/e2e_framework", { data: tiny });
  expect(put.ok(), await put.text()).toBeTruthy();
  try {
    await page.goto("/");
    await expect(page.locator(".block-node").first()).toBeVisible();
    await fileMenu(page, "open-project");
    await page.locator(".issue", { hasText: "e2e_framework" }).click();
    await expect(page.getByTestId("status")).toHaveText("No problems", { timeout: 30_000 });
    await page.getByRole("button", { name: "Training", exact: true }).click();
    const framework = page.getByTestId("framework");
    await expect(framework.locator("option")).toHaveText(["Auto (PyTorch)", "PyTorch", "Keras"]);
    await framework.selectOption("keras");

    await page.locator(".block-node", { hasText: "Adam" }).first().click();
    const started = page.waitForResponse((r) => r.url().endsWith("/api/lr-find") && r.ok());
    await page.getByTestId("find-lr").click();
    const { run_id } = await (await started).json();
    await expect(page.getByTestId("lr-summary")).toContainText("Suggested", { timeout: 200_000 });
    const run = await (await request.get(`/api/runs/${run_id}`)).json();
    expect(run.framework).toBe("keras");
  } finally {
    await request.delete("/api/projects/e2e_framework");
  }
});
