from pathlib import Path
import os

ROOT = Path(
    os.path.abspath(__file__)
).parent

DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

TINYIMGNET = DATA / "tiny-imgnet"
TINYIMGNET.mkdir(parents=True, exist_ok=True)

MNIST = DATA / "mnist"
MNIST.mkdir(parents=True, exist_ok=True)

CIFAR10 = DATA / "cifar10"
CIFAR10.mkdir(parents=True, exist_ok=True)

RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

FIGS = ROOT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)