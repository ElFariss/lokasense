#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET = BASE_DIR / "03_signal_model" / "dataset" / "gemini_label.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
