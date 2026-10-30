import { expect, test } from "@playwright/test";

import { fileMenu } from "./helpers";

// UI refinement (3.3): command palette, find, shortcuts, File menu and recent projects,
// in-app dialogs, the toast queue, the restore banner, the tour and keyboard access.
test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => { localStorage.removeItem("aime.autosave"); localStorage.removeItem("aime.recent"); });
  await page.reload();
  await expect(page.locator(".block-node").first()).toBeVisible();
});

test("command palette runs commands and finds blocks", async ({ page }) => {
  await page.keyboard.press("ControlOrMeta+k");
  const palette = page.getByTestId("palette");
  await expect(palette).toBeVisible();
  await palette.getByRole("combobox").fill("theme");
  const before = await page.evaluate(() => document.documentElement.dataset.theme);
  await page.keyboard.press("Enter");
  await expect(palette).toHaveCount(0);
  await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).not.toBe(before);

  // find: Ctrl/⌘ F lists the canvas blocks; Enter selects one
  await page.locator(".react-flow__pane").click({ position: { x: 5, y: 5 } });
  await page.keyboard.press("ControlOrMeta+f");
  await expect(palette).toBeVisible();
  await palette.getByRole("combobox").fill("@dense");
  await expect(palette.getByRole("option").first()).toContainText(/Dense/i);
  await page.keyboard.press("Enter");
  await expect(page.locator(".block-node.selected")).toHaveCount(1);
  await expect(page.locator(".block-node.selected")).toContainText(/Dense/i);

  // nothing matches: a message, Escape closes
  await page.keyboard.press("ControlOrMeta+k");
  await palette.getByRole("combobox").fill("zzzz nothing");
  await expect(palette).toContainText("No command matches");
  await page.keyboard.press("Escape");
  await expect(palette).toHaveCount(0);
});

test("shortcuts sheet, select all, undo and redo from the keyboard", async ({ page }) => {
  await page.locator(".react-flow__pane").click({ position: { x: 5, y: 5 } });
  await page.keyboard.press("?");
  const sheet = page.getByTestId("shortcuts");
  await expect(sheet).toContainText("Command palette");
  await expect(sheet).toContainText("Find a block");
  await page.keyboard.press("Escape");
  await expect(sheet).toHaveCount(0);

  const count = await page.locator(".block-node").count();
  await page.keyboard.press("ControlOrMeta+a");
  await expect(page.locator(".block-node.selected")).toHaveCount(count);
  await page.keyboard.press("Delete");
  await expect(page.locator(".block-node")).toHaveCount(0);
  await page.keyboard.press("ControlOrMeta+z");
  await expect(page.locator(".block-node")).toHaveCount(count);
  await page.keyboard.press("ControlOrMeta+Shift+z");
  await expect(page.locator(".block-node")).toHaveCount(0);
  await page.keyboard.press("ControlOrMeta+z");
  // typing in a field keeps Ctrl/⌘ A for the field
  await page.getByLabel("Search blocks").fill("relu");
  await page.getByLabel("Search blocks").press("ControlOrMeta+a");
  await expect(page.locator(".block-node.selected")).toHaveCount(0);
});

test("the library works from the keyboard", async ({ page }) => {
  const count = await page.locator(".block-node").count();
  const search = page.getByLabel("Search blocks");
  await search.fill("dropout");
  await search.press("Enter");
  await expect(page.locator(".block-node")).toHaveCount(count + 1);
  await search.fill("");
  const category = page.locator(".category > button").first();
  const expanded = await category.getAttribute("aria-expanded");
  await category.click();
  await expect(category).toHaveAttribute("aria-expanded", expanded === "true" ? "false" : "true");
  await search.fill("relu");
  const item = page.locator(".block-item").first();
  await item.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator(".block-node")).toHaveCount(count + 2);
});

test("File menu: save, recent projects, export PNG", async ({ page }) => {
  await page.locator(".topbar .project-name").fill("e2e-recent");
  await page.keyboard.press("ControlOrMeta+s");
  await expect(page.getByRole("status")).toContainText("Saved “e2e-recent”");
  await page.getByTestId("file-menu").click();
  const menu = page.getByRole("menu", { name: "File" });
  await expect(menu.getByRole("group", { name: "Recent projects" })).toContainText("e2e-recent");
  await page.keyboard.press("Escape");
  await expect(menu).toHaveCount(0);

  const download = page.waitForEvent("download");
  await fileMenu(page, "Export canvas as PNG");
  const file = await download;
  expect(file.suggestedFilename()).toBe("e2e-recent.png");
  const bytes = await (await file.createReadStream()).toArray();
  const png = Buffer.concat(bytes);
  expect(png.subarray(1, 4).toString()).toBe("PNG");
  expect(png.length).toBeGreaterThan(2000);

  // open it again from Recent
  await page.locator(".topbar .project-name").fill("changed");
  await page.getByTestId("file-menu").click();
  await menu.getByRole("menuitem", { name: "e2e-recent" }).click();
  await expect(page.locator(".topbar .project-name")).toHaveValue("e2e-recent");
});

test("in-app confirm, and toasts queue up", async ({ page }) => {
  await page.getByTestId("open-backends").click();
  const dialog = page.getByRole("dialog", { name: "Backends: where runs train" });
  await dialog.getByLabel("Name").fill("e2e-confirm");
  await dialog.getByLabel("Kind").selectOption("colab");
  await dialog.getByLabel("manual").fill("true");
  await dialog.getByTestId("save-backend").click();
  await expect(dialog.getByTestId("backends")).toContainText("e2e-confirm");
  await dialog.getByRole("row", { name: /e2e-confirm/ }).getByRole("button", { name: "Remove" }).click();
  const ask = page.getByTestId("ask-confirm");
  await expect(ask).toContainText("Remove the backend e2e-confirm");
  await page.keyboard.press("Escape");                  // closes the question, not the dialog
  await expect(ask).toHaveCount(0);
  await expect(dialog.getByTestId("backends")).toContainText("e2e-confirm");
  await dialog.getByRole("row", { name: /e2e-confirm/ }).getByRole("button", { name: "Remove" }).click();
  await ask.getByRole("button", { name: "Remove" }).click();
  await expect(dialog.getByTestId("backends")).not.toContainText("e2e-confirm");
  await dialog.getByRole("button", { name: "Close" }).click();

  // two different messages stack; the same message again is counted, × closes one
  await page.locator(".topbar .project-name").fill("e2e-toasts");
  await page.keyboard.press("ControlOrMeta+s");
  await expect(page.getByTestId("toast").filter({ hasText: "Saved “e2e-toasts”" })).toBeVisible();
  await page.keyboard.press("ControlOrMeta+s");
  await page.locator(".react-flow__pane").click({ position: { x: 5, y: 5 } });
  await page.keyboard.press("ControlOrMeta+k");
  await page.getByTestId("palette").getByRole("combobox").fill("group selected");
  await page.keyboard.press("Enter");
  const toasts = page.getByTestId("toast");
  await expect(toasts.filter({ hasText: "Saved “e2e-toasts”" })).toContainText("×2");
  await expect(toasts.filter({ hasText: "Select the blocks to group first" })).toBeVisible();
  const n = await toasts.count();
  await toasts.first().getByRole("button", { name: "Dismiss" }).click();
  await expect(toasts).toHaveCount(n - 1);
});

test("restored designs can be discarded; the tour shows once", async ({ page }) => {
  await page.locator(".block-node").first().click();
  await page.keyboard.press("Delete");
  const kept = await page.locator(".block-node").count();
  // wait for the autosave itself rather than a fixed sleep: the debounce is 800 ms
  await expect.poll(() => page.evaluate(() =>
    (JSON.parse(localStorage.getItem("aime.autosave") ?? "{}").nodes ?? []).length),
    { timeout: 5000 }).toBe(kept);
  await page.reload();
  const banner = page.getByTestId("restored");
  await expect(banner).toBeVisible();
  await expect(page.locator(".block-node")).toHaveCount(kept);
  await banner.getByRole("button", { name: "Start fresh" }).click();
  await expect(banner).toHaveCount(0);
  await expect(page.locator(".block-node")).toHaveCount(kept + 1);

  await page.evaluate(() => localStorage.setItem("aime.tour", "show"));
  await page.reload();
  const tour = page.getByTestId("tour");
  await expect(tour).toContainText("Step 1 of 3");
  await tour.getByRole("button", { name: "Next" }).click();
  await expect(tour).toContainText("Wire them up");
  await tour.getByRole("button", { name: "Next" }).click();
  await tour.getByRole("button", { name: "Done" }).click();
  await expect(tour).toHaveCount(0);
  await page.reload();
  await expect(page.locator(".block-node").first()).toBeVisible();
  await expect(page.getByTestId("tour")).toHaveCount(0);
  // and back from the palette
  await page.keyboard.press("ControlOrMeta+k");
  await page.getByTestId("palette").getByRole("combobox").fill("tour");
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("tour")).toBeVisible();
});
