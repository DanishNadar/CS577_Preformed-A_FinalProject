# Wrapper for python -m src.ablations.runAblations (same arguments)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ablations.runAblations import main  # noqa: E402

if __name__ == "__main__":
    main()
