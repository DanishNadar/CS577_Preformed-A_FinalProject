from __future__ import annotations

import random

import numpy as np
import torch


def seedEverything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seedDataLoaderWorker(workerId: int) -> None:
    # Each worker derives its NumPy/random seed from the torch seed PyTorch assigned it
    workerSeed = torch.initial_seed() % 2**32
    np.random.seed(workerSeed)
    random.seed(workerSeed)


def resolveDevice(requestedDevice: str = "auto") -> torch.device:
    if requestedDevice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requestedDevice.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warning] device '{requestedDevice}' requested but CUDA is unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(requestedDevice)


def captureRngState() -> dict:
    rngState = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rngState["cuda"] = torch.cuda.get_rng_state_all()
    return rngState


def restoreRngState(rngState: dict) -> None:
    random.setstate(rngState["python"])
    np.random.set_state(rngState["numpy"])
    torch.set_rng_state(rngState["torch"])
    if "cuda" in rngState and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rngState["cuda"])
