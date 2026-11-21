#!/usr/bin/env python3
"""Compatibility entrypoint; use train_wise_ronin.py for all current options."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train_wise_ronin.py")), run_name="__main__")
