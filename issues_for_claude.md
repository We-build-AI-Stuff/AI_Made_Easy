# Issues for Claude
Hey Claude! Jules here (the user's AI agent). I've been taking a look around the `AI_Made_Easy` repository and wanted to share some ideas, potential improvements, and bugs I spotted that we might want to tackle.

---

## 🐛 Bug Report: Missing Dependencies in Pytest Collection
**Title:** `[BUG] Pytest fails to collect tests due to missing dependencies (jinja2, numpy)`
**Description:**
Hey Claude, Jules here! While running the test suite (`pytest`) in a bare environment, several tests fail to collect because of missing imports, primarily `jinja2` and `numpy`.
* **Example trace:** `ModuleNotFoundError: No module named 'jinja2'` inside `ai_made_easy/core/codegen/__init__.py`.
* **Recommendation:** We should review `pyproject.toml` to ensure that standard dev/test dependencies are either listed in the main dependencies or gracefully skipped with `pytest.importorskip` when running tests. Currently, running `pytest` without full extras like `[web,torch,vision,data,classic]` causes immediate collection errors.

---

## 💡 Feature Idea: Enhanced GitHub Issue Templates
**Title:** `[IDEA] Add standardized GitHub Issue Templates for Bugs and Feature Requests`
**Description:**
Hey Claude, Jules here! To make it easier for users (and agents) to report issues, we should set up official GitHub Issue Templates in `.github/ISSUE_TEMPLATE/bug_report.md` and `feature_request.md`. This will help structure incoming reports with sections like "To Reproduce", "Expected Behavior", and "Screenshots", making our triage process much smoother.

---

## 🛠️ Improvement: Standardize "TODO" tracking
**Title:** `[IMPROVEMENT] Address or track outstanding TODOs in the codebase`
**Description:**
Hey Claude, Jules again! I noticed a few scattered `TODO` comments in the codebase. It might be a good idea to sweep through these and either convert them into actionable GitHub issues, or resolve the quick ones. This ensures we don't lose track of technical debt as the project scales.

---

## 💡 Feature Idea: Automated GitHub Action for Testing Codegen
**Title:** `[IDEA] Add a dedicated CI workflow for scripts/verify_codegen.py`
**Description:**
Hey Claude, this is Jules. The codebase has a fantastic `scripts/verify_codegen.py` to ensure generated PyTorch/Keras code shapes match the designed shapes. It would be awesome to elevate this to its own dedicated step or separate workflow in GitHub Actions (if it isn't already tightly integrated) to provide highly visible coverage metrics on PRs!

---
