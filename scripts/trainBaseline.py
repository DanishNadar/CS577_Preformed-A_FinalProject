"""Train the baseline. Same arguments as `python -m src.training.train`.

    python scripts/trainBaseline.py --config configs/baseline.yaml
    python scripts/trainBaseline.py --set training.epochs=5 --set device=cuda
    python scripts/trainBaseline.py --resume checkpoints/baseline/last.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train import main  # noqa: E402

if __name__ == "__main__":
    main()
