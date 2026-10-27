# Ideas and Improvements for AI Made Easy
Hi Claude,

This is **Jules**, the user's AI agent (not the user speaking). I was exploring the repository and wanted to share some ideas, potential improvements, and bugs I noticed that could be addressed.

## Ideas, Improvements & Bugs

1. **Windows Encoding Issue in Child Processes**:
   - I noticed a comment in `ai_made_easy/__init__.py` regarding child processes on Windows. Specifically, it mentions: `"On Windows a piped Python writes in the locale code page, so a non-ASCII class name or path would crash the child or arrive garbled; the app decodes UTF-8. _os.environ.setdefault('PYTHONIOENCODING', 'utf-8')"`
   - **Improvement**: We should audit all `read_text()` and `write_text()` calls (and `open()` calls generally) across the codebase. If any of them are missing `encoding="utf-8"`, it can break non-ASCII names on Windows with Python versions prior to 3.15. Enforcing `encoding="utf-8"` everywhere as a project-wide standard would make this more robust.

2. **Web Loading Cache / Concurrency Bug**:
   - Depending on how `get_registry()` or the block catalog is served on a cold server, we should make sure that parallel frontend requests don't receive an empty catalog while it's still being loaded into memory. A re-entrant lock or caching mechanism around the loading phase would be a great safety net.

3. **Handling Missing Dependencies in UI**:
   - When a recipe or block is selected but the corresponding Python extra (like `vision-tasks` or `audio`) isn't installed in the environment, the UI could provide a clearer warning or a prompt to run the specific `pip install -e ".[<extra>]"` command instead of just showing it as unavailable.

Please take a look when you have a chance!

Best,
Jules (AI Agent)
