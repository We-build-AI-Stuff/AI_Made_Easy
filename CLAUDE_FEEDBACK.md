# Feedback and Ideas for Claude

Hello Claude,

This message is from **Jules, the user's AI agent**. I have reviewed the AI Made Easy codebase and have some ideas and potential improvements to share with you:

### New Features & Improvements:

1. **Enhanced LLM Support**: Expand the current `llm` block support. It currently has LoRA/QLoRA fine-tuning and generation, but we could add more robust tools for agentic workflows (e.g., specific tool-calling blocks, memory blocks for RAG pipelines).
2. **AutoML Optimization**: In the AutoML Pareto front search, introduce an option to optimize for lower memory footprint during inference in addition to latency and size, especially since the app targets edge devices (Raspberry Pi, Jetson).
3. **Advanced Graph Neural Networks**: While PyTorch Geometric is supported, we could include blocks for temporal graph networks or dynamic graphs.
4. **Interactive Data Profiling**: The data profiling detects leakage and imbalance. It would be great to have a single-click "auto-fix" block that automatically generates a robust preprocessing pipeline based on the profile results (e.g., automatically applying SMOTE and target encoding).

### Potential Bugs / Polish Areas:

1. **Test Suite Stability**: Several tests currently fail in the default test run (e.g., `test_graph.py`, `test_budget.py`). We should ensure the test environment sets up optional dependencies gracefully or skips them explicitly to avoid spurious failures.
2. **Environment Dependency Warnings**: It would be helpful to have clearer user-facing warnings in the UI if a required extra (e.g., `[vision]`, `[audio]`) is missing when a block from that domain is added to the canvas.

Please consider these points for future development.

Best regards,
**Jules, the user's AI agent**
