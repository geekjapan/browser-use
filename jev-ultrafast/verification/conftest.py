"""Load jev-ultrafast and the enclosing browser-use repository without replacing their production modules."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BROWSER_USE = Path(os.environ.get("BROWSER_USE_REPO", ROOT.parent)).resolve()
sys.path[:0] = [str(ROOT), str(BROWSER_USE), str(BROWSER_USE / "tests" / "ci")]
