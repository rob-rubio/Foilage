"""Entry point: python -m foilage_gui [input.json]"""

from pathlib import Path
import sys

from .cli import main

if __name__ == "__main__":
    main()
