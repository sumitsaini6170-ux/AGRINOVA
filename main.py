
"""
AGRINOVA - AI-Powered Smart Farming Assistant
Main application launcher
Developer: Sumit Kumar
"""

import runpy
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    bridge_file = BASE_DIR / "backend" / "bridge.py"

    if not bridge_file.exists():
        raise FileNotFoundError(
            f"Bridge file not found: {bridge_file}"
        )

    runpy.run_path(
        str(bridge_file),
        run_name="__main__"
    )
