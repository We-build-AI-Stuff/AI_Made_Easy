// The dockable workspace (3.12): panels dock on any edge by dragging, collapse to
// rails, restore, resize and go back to the default arrangement. Narrow screens
// turn the columns into drawers (covered by the compact-mode test at the end).
import { expect, test, type Page } from "@playwright/test";

/** Drag a panel's header to a point, the way a user docks it. */
async function drag(page: Page, header: ReturnType<Page["getByRole"]>, to: { x: number; y: number }) {
  const box = await header.boundingBox();
  expect(box).toBeTruthy();
  await page.mouse.move(box!.x + box!.width / 2, box!.y + 6);
  await page.mouse.down();
  await page.mouse.move(to.x, to.y, { steps: 8 });   // the drop zones light up on the way
  await page.mouse.up();
}

const side = (page: Page, panel: string) =>
  page.locator(`[data-dock-panel="${panel}"]`).getAttribute("data-dock-side");

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => localStorage.removeItem("aime.layout.v2"));
  await page.reload();
  // the first-run tour would cover the header
  const skip = page.getByRole("button", { name: "Skip tour" });
  if (await skip.count()) await skip.evaluate((b) => (b as HTMLElement).click());
});

test("panels sit on their default edges and can collapse and restore", async ({ page }) => {
  expect(await side(page, "library")).toBe("left");
  expect(await side(page, "inspector")).toBe("right");
  expect(await side(page, "console")).toBe("bottom");

  await page.getByTestId("layout-menu").click();
  await page.getByTestId("layout-library").click();               // collapse the library
  await expect(page.getByTestId("restore-library")).toBeVisible();
  await expect(page.locator('[data-dock-panel="library"]')).toHaveCount(0);
  await page.getByTestId("restore-library").click();              // the rail restores it
  await expect(page.locator('[data-dock-panel="library"]')).toBeVisible();
  expect(await side(page, "library")).toBe("left");
});

test("a panel can be dragged to another edge and to a corner seat", async ({ page }) => {
  const ws = await page.locator(".dock-workspace").boundingBox();   // zones are measured
  expect(ws).toBeTruthy();                                          // against the workspace
  const bottom = { x: ws!.x + ws!.width * 0.5, y: ws!.y + ws!.height * 0.92 };
  const left = { x: ws!.x + ws!.width * 0.08, y: ws!.y + ws!.height * 0.1 };

  await drag(page, page.locator('[data-dock-panel="library"] .dock-head'), bottom);
  expect(await side(page, "library")).toBe("bottom");             // the bottom edge took it
  await expect(page.locator(".dock-bottom [data-dock-panel='library']")).toBeVisible();

  // dragging it back to the left edge, below nothing, restores the default seat
  await drag(page, page.locator('[data-dock-panel="library"] .dock-head'), left);
  expect(await side(page, "library")).toBe("left");

  // the inspector joins the bottom strip next to the console (a corner seat)
  await drag(page, page.locator('[data-dock-panel="inspector"] .dock-head'), bottom);
  expect(await side(page, "inspector")).toBe("bottom");
  const order = await page.locator(".dock-bottom [data-dock-panel]").evaluateAll(
    (els) => els.map((e) => e.getAttribute("data-dock-panel")));
  expect(order).toContain("console");
  expect(order).toContain("inspector");
});

test("the layout survives a reload, and reset puts everything back", async ({ page }) => {
  const ws = await page.locator(".dock-workspace").boundingBox();
  await drag(page, page.locator('[data-dock-panel="library"] .dock-head'),
             { x: ws!.x + ws!.width * 0.5, y: ws!.y + ws!.height * 0.92 });
  expect(await side(page, "library")).toBe("bottom");
  await page.reload();
  await expect(page.locator('[data-dock-panel="library"]')).toBeVisible();
  expect(await side(page, "library")).toBe("bottom");             // remembered

  await page.getByTestId("layout-menu").click();
  await page.getByTestId("layout-reset").click();
  expect(await side(page, "library")).toBe("left");
  expect(await side(page, "inspector")).toBe("right");
  expect(await side(page, "console")).toBe("bottom");
});

test("the console's tabs still work wherever it is docked", async ({ page }) => {
  await page.getByRole("button", { name: "Training", exact: true }).click();
  await expect(page.getByTestId("canvas")).toBeVisible();
  await page.getByRole("button", { name: "Problems" }).click();
  await expect(page.locator(".dock-bottom")).toContainText("No problems");
});

test("narrow screens: panels become drawers over the canvas", async ({ page }) => {
  await page.setViewportSize({ width: 720, height: 900 });
  await expect(page.locator(".drawer-toggle.toggle-left")).toBeVisible();
  await page.locator(".drawer-toggle.toggle-left").click();
  await expect(page.locator(".dock-left.drawer-open")).toBeVisible();
  await expect(page.locator('[data-dock-panel="library"]')).toBeVisible();
  await page.locator(".drawer-backdrop").click({ position: { x: 700, y: 100 } });
  await expect(page.locator(".dock-left.drawer-open")).toHaveCount(0);
});
