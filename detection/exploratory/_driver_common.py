"""Shared bits for the exploratory drivers: worker cap and run provenance."""

from __future__ import annotations

import os
import platform
import subprocess

MAX_WORKERS = 8


def provenance() -> dict:
    """Commit, dirty flag and library versions, written into every summary."""
    here = os.path.dirname(os.path.abspath(__file__))

    def git(*a):
        try:
            return subprocess.run(["git", "-C", here, *a], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    import numpy
    import pandas
    import scipy
    import sklearn
    return {
        "git_commit": git("rev-parse", "HEAD"),
        # outputs written under _dev_smoke/ do not count as a dirty tree
        "git_dirty": bool(git("status", "--porcelain", "--", "..",
                              ":(exclude)_dev_smoke")),
        "python": platform.python_version(),
        "numpy": numpy.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "pandas": pandas.__version__,
    }
