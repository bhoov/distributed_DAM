#%%
import datasets
from datasets import load_dataset
import numpy as np
from scipy.io.arff import loadarff
import functools as ft
import path_fixes as pf
from enum import Enum


def parse_arff(dataset):
    data = []
    targets = []
    for row in dataset:
        row = tuple(row)
        data.append(row[:-1])
        targets.append(row[-1])
    return np.array(data), np.array(targets)

@ft.lru_cache
def get_letter_data():
    dataset, meta = loadarff(str(pf.DATA / 'letter.arff'))
    data, targets = parse_arff(dataset)
    return data, targets

@ft.lru_cache
def get_eyestate_data():
    dataset, meta = loadarff(str(pf.DATA / 'eeg-eye-state.arff'))
    data, targets = parse_arff(dataset)
    return data, targets

@ft.lru_cache
def get_phoneme_data():
    dataset, meta = loadarff(str(pf.DATA / 'phoneme.arff'))
    data, targets = parse_arff(dataset)
    return data, targets

def prep_mnist_data():
    # Check if Xtrain.npy exists
    if not (pf.MNIST / "Xtrain.npy").exists():
        print("Prepping MNIST data...")
        ds = load_dataset('mnist').with_format('np')
        Xtrain = ds['train']['image']
        Ytrain = ds['train']['label']
        Xtrain = Xtrain / 255.

        Xtest = ds['test']['image']
        Ytest = ds['test']['label'] 
        Xtest = Xtest / 255.

        np.save(pf.MNIST / "Xtrain.npy", Xtrain)
        np.save(pf.MNIST / "Ytrain.npy", Ytrain)
        np.save(pf.MNIST / "Xtest.npy", Xtest)
        np.save(pf.MNIST / "Ytest.npy", Ytest)
    else:
        print("MNIST data already exists")

# import datasets
# desired_shape = (64, 64, 3)
# Xtrain = np.stack([im for im in tinyimnet['train']['image'] if im.shape == desired_shape])
# Xtest = np.stack([im for im in tinyimnet['valid']['image'] if im.shape == desired_shape])
# Xtrain = Xtrain / 255.
# Xtest = Xtest / 255.
# np.save(pf.TINYIMGNET/"Xtrain.npy", Xtrain)
# np.save(pf.TINYIMGNET/"Xtest.npy", Xtest)


# import datasets
# cifar = datasets.load_dataset("cifar10").with_format("jax")
# Xtrain = cifar['train']['img']
# Ytrain = cifar['train']['label']
# Xtest = cifar['test']['img']
# Ytest = cifar['test']['label']

# Xtrain = Xtrain / 255.
# Xtest = Xtest / 255.

# np.save(pf.CIFAR10/"Xtrain.npy", Xtrain)
# np.save(pf.CIFAR10/"Ytrain.npy", Ytrain)
# np.save(pf.CIFAR10/"Xtest.npy", Xtest)
# np.save(pf.CIFAR10/"Ytest.npy", Ytest)

# Xtrain = np.load(pf.CIFAR10/"Xtrain.npy")
# Ytrain = np.load(pf.CIFAR10/"Ytrain.npy")
# Xtest = np.load(pf.CIFAR10/"Xtest.npy")
# Ytest = np.load(pf.CIFAR10/"Ytest.npy")

def prep_cifar_data():
    # Check if Xtrain.npy exists
    if not (pf.CIFAR10 / "Xtrain.npy").exists():
        print("Prepping CIFAR data...")
        cifar = datasets.load_dataset("cifar10").with_format("jax")
        Xtrain = cifar['train']['img']
        Ytrain = cifar['train']['label']
        Xtest = cifar['test']['img']
        Ytest = cifar['test']['label']
        Xtrain = Xtrain / 255.
        Xtest = Xtest / 255.

        np.save(pf.CIFAR10/"Xtrain.npy", Xtrain)
        np.save(pf.CIFAR10/"Ytrain.npy", Ytrain)
        np.save(pf.CIFAR10/"Xtest.npy", Xtest)
        np.save(pf.CIFAR10/"Ytest.npy", Ytest)

    else:
        print("CIFAR data already exists")

def prep_tinyimnet_data():
    # Check if Xtrain.npy exists
    if not (pf.TINYIMGNET / "Xtrain.npy").exists():
        print("Prepping TinyImgnet data...")
        ds = datasets.load_dataset("zh-plus/tiny-imagenet").with_format("jax")
        desired_shape = (64, 64, 3)
        Xtrain = np.stack([im for im in ds['train']['image'] if im.shape == desired_shape])
        Xtest = np.stack([im for im in ds['valid']['image'] if im.shape == desired_shape])
        Xtrain = Xtrain / 255.
        Xtest = Xtest / 255.
        np.save(pf.TINYIMGNET/"Xtrain.npy", Xtrain)
        np.save(pf.TINYIMGNET/"Xtest.npy", Xtest)
    else:
        print("TinyImgnet data already exists")

def get_mnist_traindata():
    Xtrain, Ytrain = np.load(pf.MNIST / "Xtrain.npy"), np.load(pf.MNIST / "Ytrain.npy")
    return Xtrain, Ytrain

def get_mnist_testdata():
    Xtest, Ytest = np.load(pf.MNIST / "Xtest.npy"), np.load(pf.MNIST / "Ytest.npy")
    return Xtest, Ytest

def get_cifar_traindata():
    Xtrain, Ytrain = np.load(pf.CIFAR10 / "Xtrain.npy"), np.load(pf.CIFAR10 / "Ytrain.npy")
    return Xtrain, Ytrain

def get_cifar_testdata():
    Xtest, Ytest = np.load(pf.CIFAR10 / "Xtest.npy"), np.load(pf.CIFAR10 / "Ytest.npy")
    return Xtest, Ytest

def get_tiny_imagenet_traindata():
    Xtrain = np.load(pf.TINYIMGNET / "Xtrain.npy")
    Ytrain = None
    return Xtrain, Ytrain

def get_tiny_imagenet_testdata():
    Xtest = np.load(pf.TINYIMGNET / "Xtest.npy")
    Ytest = None
    return Xtest, Ytest

class DataOpts(Enum):
    letter = "letter"
    phoneme = "phoneme"
    eyestate = "eyestate"
    mnist = "mnist"
    cifar10 = "cifar10"
    tiny_imagenet = "tiny_imagenet"
    def __repr__(self): return self.value
    def __str__(self): return self.value

def get_data(datatype: DataOpts = DataOpts.mnist):
    """Return training data for a specified dataset, with minimum adjusted to be 0"""
    if datatype == DataOpts.letter:
        return get_letter_data()[0]
    if datatype == DataOpts.phoneme:
        Xtrain = get_phoneme_data()[0]
        Xmin = Xtrain.min()
        return Xtrain - Xmin
    if datatype == DataOpts.eyestate:
        Xtrain = get_eyestate_data()[0]
        Xmin = Xtrain.min()
        return Xtrain - Xmin
    if datatype == DataOpts.mnist:
        return get_mnist_traindata()[0]
    if datatype == DataOpts.cifar10:
        return get_cifar_traindata()[0]
    if datatype == DataOpts.tiny_imagenet:
        return get_tiny_imagenet_traindata()[0]
    raise ValueError(f"Unknown datatype: {datatype}")


if __name__ == "__main__":
    ldata, ltargets = get_letter_data() # data is (20_000, 16)
    # edata, etargets = get_eyestate_data() # data is (14980, 140)
    # pdata, ptargets = get_phoneme_data() # data is (5404, 5)
    prep_mnist_data()
    prep_cifar_data()
    prep_tinyimnet_data()
