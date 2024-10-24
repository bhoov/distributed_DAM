#%%
"""Experiment adapted from ALN: alignment experiments

# kernels = ["CosL2DAM", "SinCosL2DAM"]
python exp_QUANT2__kernel_ablations.py --add_bias --outdir results/QUANT2-manybeta
"""
# %reload_ext autoreload
# %autoreload 2
#%%
import numpy as np
import jax.numpy as jnp
import jax
import jax.random as jr
import matplotlib.pyplot as plt
from einops import rearrange
import path_fixes as pf
from pathlib import Path
from tools import outerize
from data_utils import get_letter_data, get_eyestate_data, get_phoneme_data, get_mnist_traindata
from dataclasses import dataclass
from enum import Enum
from plotting import show_img
import tyro
from typing import *
import os
import equinox as eqx
import functools as ft
import jax.tree_util as jtu
from jaxtyping import Float, Array
import pandas as pd
from tqdm.auto import tqdm, trange

from kernel_sims import SIM_REGISTRY as KernelOpts

class DataOpts(Enum):
    letter = "letter"
    phoneme = "phoneme"
    eyestate = "eyestate"
    mnist = "mnist"

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value

def get_data(datatype: DataOpts = DataOpts.letter):
    if datatype == DataOpts.letter: return get_letter_data()[0]
    elif datatype == DataOpts.phoneme: return get_phoneme_data()[0]
    elif datatype == DataOpts.eyestate: return get_eyestate_data()[0]
    elif datatype == DataOpts.mnist: return get_mnist_traindata()[0]
    else: raise ValueError(f"Unknown datatype: {datatype}")

@dataclass
class Args:
    """Compare the kernelized DAM energy, similarity, and updates to the standard DAM"""
    outdir: Optional[Union[Path,str]] = None # Where to save the results
    data: DataOpts = DataOpts.letter # Which dataset to evaluate on
    kernels: Tuple[str] = ("CosL2DAM", "ExpL2DAM", "ExpExpL2DAM", "SinCosL2DAM") # Which kernel to use, specified by the KernelOpts registry. Ignore "nolog" vs "log", both are tested
    seed: int = 42 # Random seed for all experiments
    add_bias: bool = True # If true, add b ~ U(0,2pi) to the basis function's argument
    orthogonal_init: bool = False # If true, initialize random features from orthogonal gaussian
    device: str = "" # Which device to use. CPU by default, or int to specify which CUDA device
    betas: Tuple[Union[float, str]] = (1., 5., 10., 20., 30., 40., 50., 60., 100., 200., 600.) # Specify the beta values to use. Can be a float, or "opt" to indicate using the optimal beta
    ms: Tuple[int] = (40000, 80000, 160000, 275000, 400000, 800000) # Specify which values of `m` to test
    nmemories: int = 500 # Number of memories to store
    nqueries: int = 400 # Number of queries to use

    def get_outdir(self):
        if self.outdir is None:
            outdir = pf.RESULTS / f"QUANT2--data={self.data}{'_nobias' if not self.add_bias else ''}{'_orth' if self.orthogonal_init else ''}"
        else:
            outdir = self.outdir
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        return outdir

    def get_ckptdir(self):
        ckpt_dir = self.get_outdir() / "ckpts"
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        return ckpt_dir

    def get_rng0(self):
        return jr.PRNGKey(self.seed)

@dataclass
class ExpRecord:
    data: str 
    nqueries: int
    nmemories: int
    beta: float 
    m: int
    kernel: str
    queries: Float[Array, "q d"]
    memories: Float[Array, "N d"]
    add_bias: bool

    Emems: Float[Array, "N"]
    logEmems: Float[Array, "N"]
    gradEmems: Float[Array, "N d"]
    gradlogEmems: Float[Array, "N d"]
    Equeries: Float[Array, "q"]
    logEqueries: Float[Array, "q"]
    gradEqueries: Float[Array, "q d"]
    gradlogEqueries: Float[Array, "q d"]

    Emems_kernel: Float[Array, "N"]
    logEmems_kernel: Float[Array, "N"]
    gradEmems_kernel: Float[Array, "N d"]
    gradlogEmems_kernel: Float[Array, "N d"]
    Equeries_kernel: Float[Array, "q"]
    logEqueries_kernel: Float[Array, "q"]
    gradEqueries_kernel: Float[Array, "q d"]
    gradlogEqueries_kernel: Float[Array, "q d"]

    def to_dict(self): return self.__dict__
    def to_series(self): return pd.Series(self.to_dict())

#%%
class Normalizer:
    """Normalize the data s.t. the maximum L2 distance between two points is 1.0"""
    def __init__(self, d, minval=0., maxval=1):
        self.minval = minval
        self.maxval = maxval
        self.d = d
        self.range = self.maxval - self.minval

    @classmethod
    def from_data(cls, M: Float[Array, "... d"]):
        d = M.shape[-1]
        return cls(d, M.min(), M.max())

    def __call__(self, M):
        return (M - self.minval) / (self.range * np.sqrt(self.d))

## Start Main
args = tyro.cli(Args)
# args = Args(add_bias=False)
rng = args.get_rng0()

kernels = sorted(args.kernels)
betas = sorted(args.betas, key=lambda x: np.inf if isinstance(x, str) else x, reverse=True)
ms = sorted(args.ms, reverse=True)

# Process data
M = get_data(args.data)
M = rearrange(M, "n ... -> n (...)")
normalizer = Normalizer.from_data(M)
M = normalizer(M)
N, d = M.shape

k, rng = jr.split(rng)
idxs = jr.choice(k, N, shape=(args.nmemories + args.nqueries,), replace=False)

memories = M[idxs[:args.nmemories]]
queries = M[idxs[args.nmemories:]]


#%%
def islogdam(kname:str) -> bool:
    return not "nolog" in kname.lower()

def logify(kname):
    if islogdam(kname): return kname
    return kname.replace("NoLogL2DAM", "L2DAM")

def unlogify(kname):
    if islogdam(kname): return kname.replace("L2DAM", "NoLogL2DAM")
    return kname

istep = 0
total = len(kernels) * len(ms) * len(betas)
pbar = tqdm(total=total)
for kernel in kernels:
    for m in ms:
        for beta in betas:
            istep += 1
            pbar.update(1)
            pbar.set_description(f"Step {istep}, m={m}, beta={beta}")

            # Do logify
            kname = logify(kernel)
            KClass = KernelOpts[kname]
            if isinstance(beta, str) and beta.lower() == "opt":
                print(kname)
                beta_use = (KClass(rng, d, m, beta=1., add_bias=args.add_bias, orthogonal_init=args.orthogonal_init)
                            .condition_beta_on_memories(memories)).beta
            elif isinstance(beta, float):
                beta_use = beta
            else:
                raise ValueError(f"Unknown beta value: {beta}")

            kdam = KClass(rng, d, m, beta=beta_use, add_bias=args.add_bias, orthogonal_init=args.orthogonal_init)
            venergyf = jax.vmap(jax.value_and_grad(kdam.energy), in_axes=(0, None))
            T = kdam.kernelize_memories(memories)
            venergyf_kernel = jax.vmap(jax.value_and_grad(kdam.kernel_energy), in_axes=(0, None))

            logEmems, gradlogEmems = venergyf(memories, memories)
            logEqueries, gradlogEqueries = venergyf(queries, memories)
            logEmems_kernel, gradlogEmems_kernel = venergyf_kernel(memories, T)
            logEqueries_kernel, gradlogEqueries_kernel = venergyf_kernel(queries, T)

            # Do Unlogged
            kname = unlogify(kernel)
            KClass = KernelOpts[kname]
            if isinstance(beta, str) and beta.lower() == "opt":
                print(kname)
                beta_use = (KClass(rng, d, m, beta=1., add_bias=args.add_bias, orthogonal_init=args.orthogonal_init)
                            .condition_beta_on_memories(memories)).beta
            elif isinstance(beta, float):
                beta_use = beta
            else:
                raise ValueError(f"Unknown beta value: {beta}")
            kdam = KClass(rng, d, m, beta=beta_use, add_bias=args.add_bias, orthogonal_init=args.orthogonal_init)
            venergyf = jax.vmap(jax.value_and_grad(kdam.energy), in_axes=(0, None))
            T = kdam.kernelize_memories(memories)
            venergyf_kernel = jax.vmap(jax.value_and_grad(kdam.kernel_energy), in_axes=(0, None))

            Emems, gradEmems = venergyf(memories, memories)
            Equeries, gradEqueries = venergyf(queries, memories)
            Emems_kernel, gradEmems_kernel = venergyf_kernel(memories, T)
            Equeries_kernel, gradEqueries_kernel = venergyf_kernel(queries, T)

            # Save record
            record = ExpRecord(
                data=str(args.data),
                nqueries=args.nqueries,
                nmemories=args.nmemories,
                beta=beta_use,
                m=m,
                kernel=logify(kernel),
                queries=queries,
                memories=memories,
                add_bias=args.add_bias,

                Emems=Emems,
                logEmems=logEmems,
                gradEmems=gradEmems,
                gradlogEmems=gradlogEmems,
                Equeries=Equeries,
                logEqueries=logEqueries,
                gradEqueries=gradEqueries,
                gradlogEqueries=gradlogEqueries,

                Emems_kernel=Emems_kernel,
                logEmems_kernel=logEmems_kernel,
                gradEmems_kernel=gradEmems_kernel,
                gradlogEmems_kernel=gradlogEmems_kernel,
                Equeries_kernel=Equeries_kernel,
                logEqueries_kernel=logEqueries_kernel,
                gradEqueries_kernel=gradEqueries_kernel,
                gradlogEqueries_kernel=gradlogEqueries_kernel,
            )

            fname = f"{istep:03d}--_m={m}_beta={beta}_kernel={logify(kname)}.pkl"
            outf = args.get_ckptdir() / fname
            print(f"SAVING CHECKPOINT {outf}")
            df = pd.DataFrame([record.to_dict()])
            df.to_pickle(outf)
# %%
