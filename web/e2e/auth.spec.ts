// User accounts (3.12): a server started with `aime web --auth` shows a sign-in
// screen; members see only their own projects; admins manage users. This spec
// starts its own server on a private port, with accounts prepared through
// `aime users` the way an operator would.
import { spawn as nodeSpawn, type ChildProcess } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test, type Page } from "@playwright/test";

const PORT = 8807;
const HOME = mkdtempSync(join(tmpdir(), "aime-auth-"));
const AIME = process.env.AIME_BIN ?? "aime";
let server: ChildProcess | null = null;

/** `aime users …` reads the password twice from stdin when it is not a terminal. */
function users(command: string, password: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const parts = command.split(" ");
    const child = nodeSpawn(parts[0], parts.slice(1),
                            { env: { ...process.env, AIME_HOME: HOME } });
    let err = "";
    child.stderr!.on("data", (d) => { err += d; });
    child.on("error", reject);
    child.on("exit", (code) => code === 0 ? resolve() : reject(new Error(err || `exit ${code}`)));
    child.stdin!.write(`${password}\n${password}\n`);
    child.stdin!.end();
  });
}

async function waitUp() {
  for (let i = 0; i < 60; i++) {
    try {
      if ((await fetch(`http://127.0.0.1:${PORT}/api/info`)).ok) return;
    } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`the auth server on :${PORT} did not start`);
}

test.beforeAll(async () => {
  await users(`${AIME} users add Admin --role admin`, "admin-pw");
  await users(`${AIME} users add Mia`, "mia-pw");
  await users(`${AIME} users add Bo`, "bo-pw");
  server = nodeSpawn(AIME, ["web", "--auth", "--port", String(PORT)],
                     { env: { ...process.env, AIME_HOME: HOME } });
  await waitUp();
});

test.afterAll(async () => {
  server?.kill();
  rmSync(HOME, { recursive: true, force: true });
});

async function signIn(page: Page, name: string, password: string) {
  await page.goto(`http://127.0.0.1:${PORT}/`);
  await page.getByTestId("login").waitFor();
  await page.getByLabel("Name").fill(name);
  await page.getByLabel("Password").fill(password);
  await page.getByTestId("login-submit").click();
  await expect(page.getByTestId("user-chip")).toContainText(name);
}

const projectsOf = (page: Page) =>
  page.evaluate(async (port) => {
    const res = await fetch(`http://127.0.0.1:${port}/api/projects`, {
      headers: { Authorization: `Bearer ${localStorage.getItem("aime.token")}` },
    });
    return (await res.json()).projects as { name: string; owner: string | null }[];
  }, PORT);

test("anonymous visitors must sign in; a wrong password is refused", async ({ page }) => {
  await page.goto(`http://127.0.0.1:${PORT}/`);
  await expect(page.getByTestId("login")).toBeVisible();
  await page.getByLabel("Name").fill("Admin");
  await page.getByLabel("Password").fill("wrong");
  await page.getByTestId("login-submit").click();
  await expect(page.locator(".login-error")).toContainText(/wrong name or password/i);
  await signIn(page, "Admin", "admin-pw");
});

test("an admin manages users from the browser", async ({ page }) => {
  await signIn(page, "Admin", "admin-pw");
  await page.getByTestId("open-users").click();
  await expect(page.locator('[data-testid="user-Mia"]')).toBeVisible();
  await page.getByTestId("add-user").click();
  await page.getByLabel("New user's name").fill("Zoe");
  await page.getByLabel("New user's password").fill("zoe-pw");
  await page.locator(".user-add button.primary").click();
  await expect(page.locator('[data-testid="user-Zoe"]')).toContainText("member");
});

test("members see only their own projects; admins see them all", async ({ page, browser }) => {
  const mia = await (await browser.newContext()).newPage();   // its own storage, like
  await signIn(mia, "Mia", "mia-pw");                         // another person's browser
  await expect(mia.getByTestId("open-users")).toHaveCount(0);     // members do not manage
  await mia.getByRole("banner").getByLabel("Project name").fill("mias model");
  await mia.getByRole("button", { name: "Save", exact: true }).click();
  await expect(mia.getByTestId("status")).toBeVisible();
  expect((await projectsOf(mia)).map((p) => p.name)).toContain("mias model");
  await mia.close();

  const bo = await (await browser.newContext()).newPage();
  await signIn(bo, "Bo", "bo-pw");
  expect((await projectsOf(bo)).map((p) => p.name)).not.toContain("mias model");
  await bo.close();

  await signIn(page, "Admin", "admin-pw");
  const admin = await projectsOf(page);
  expect(admin.map((p) => p.name)).toContain("mias model");
  expect(admin.find((p) => p.name === "mias model")?.owner).toBe("Mia");
});
