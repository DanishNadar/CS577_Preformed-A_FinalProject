# Train the baseline; accepts the same arguments as python -m src.training.train

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train import main  # noqa: E402

if __name__ == "__main__":
    main()
