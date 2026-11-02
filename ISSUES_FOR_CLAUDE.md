# Ideas and Issues for AI Made Easy

Hello Claude,

I am Jules, the user's AI agent. I have reviewed the AI Made Easy codebase and have some ideas for potential improvements and issues that you might want to look into or implement in future updates.

## Potential Improvements & Feature Ideas
1.  **Expand MCP Server Tools**: The current MCP server (`ai_made_easy/mcp/server.py`) has a good set of tools. Adding tools for direct graph mutation (e.g., adding/removing blocks programmatically, changing specific parameter values directly without needing to replace the whole graph) could make agents more capable of iterative design.
2.  **More LLM Providers**: The LLM blocks currently seem heavily tied to Hugging Face (`HF Model` references). Abstracting the model interface further to easily support OpenAI, Anthropic, or local Ollama instances directly as block providers (similar to LangChain integrations) could increase the utility of the LLM pipeline features.
3.  **Advanced Auto-Fixes**: The `core/fixes.py` currently handles basic corrections (like inserting Flatten layers or clamping parameters). This could be expanded to suggest entirely different architectures if latency or memory budgets fail drastically, rather than just adjusting batch size or mixed precision.
4.  **Dataset Profiling Enhancements**: In `core/data/profile.py`, handling extremely large tabular datasets might be slow. Adding sampling strategies or streaming options for the profiling step could improve responsiveness in the UI.
5.  **Agent Instruction File**: Adding an `AGENTS.md` file in the root directory might be useful to define standard procedures and conventions for any human-assisting AI agents (like me) working in this repository.

## Bugs & Edge Cases to Investigate
1.  **Windows Pipe Issue Mitigation**: The MCP server initialization (`_prepare`) has a specific workaround for Windows stdin pipe blocking. It might be worth verifying if this workaround introduces any side effects during shutdown or when the parent process terminates unexpectedly.
2.  **Lint Rule Conflicts**: In `core/lints.py`, check if any custom task loop rules conflict with the standard `PIPELINE_RULES` when a custom task is deeply nested or uses a non-standard training loop.
3.  **Export Format Fallbacks**: If a specific export format fails for a non-budget-related reason, the fallback logic in `deploy_run` could be made more robust to attempt alternative but similar formats automatically.

Best regards,
Jules, the user's AI agent
