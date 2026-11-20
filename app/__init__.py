"""Mission Replay: web demo of frame-to-map localisation (FastAPI backend + static frontend)."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # run from a checkout without `pip install -e .`
    sys.path.insert(0, str(_SRC))
