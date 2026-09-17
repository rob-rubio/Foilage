"""Foilage GUI launcher.

Usage:
    python run_gui.py [input.json] [--threads N]

Without an argument the GUI loads the newest cases/*/input.json.
"""

from foilage_gui.cli import main

if __name__ == "__main__":
    main()
