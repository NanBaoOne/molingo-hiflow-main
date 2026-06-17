"""Compatibility package for running scripts from the mogen-hiflow root.

The project keeps modules such as ``models`` and ``utils`` at this directory
level while entrypoints import them as ``mogen.models`` and ``mogen.utils``.
Expose the project root as this package's search path so those imports work
without renaming the uploaded directory.
"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

__path__ = [str(_PROJECT_ROOT)]
