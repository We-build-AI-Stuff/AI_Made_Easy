"""AI Made Easy — a visual block-based AI model builder.

Design rules that keep this scalable to agents/MCP:
  * ``ai_made_easy.core`` is pure Python (no Qt) and JSON-native.
  * ``ai_made_easy.ui`` is one frontend over core; the CLI is another.
"""
import os as _os

__version__ = "3.11.0"

# Child processes (training workers, Try it, generated scripts) talk to the app through
# pipes. On Windows a piped Python writes in the locale code page, so a non-ASCII class
# name or path would crash the child or arrive garbled; the app decodes UTF-8.
_os.environ.setdefault("PYTHONIOENCODING", "utf-8")
