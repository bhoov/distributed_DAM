"""
# Config used in QUANT1b main:
python exp_QUANT1b_opt_retrieval.py --betas 10 30 40 50 --ms 5000 40000 80000 120000 160000 200000 300000 400000 500000 --outdir results/QUANT1b_near_0.1_retrieval --do_retrieval

# For quick testing
python exp_QUANT1b_opt_retrieval.py --betas 40 50 --ms 5000 40000 --outdir results/QUANT1b_near_0.1_retrieval --do_retrieval
"""
#%%
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Float, Array, PyTree, jaxtyped
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from kernel_sims import SIM_REGISTRY as KernelOpts
from tools import outerize, init_queries_for_all_memories, binarize_data
import matplotlib.pyplot as plt
import tyro
from pathlib import Path
from typing import *
import path_fixes as pf
from tqdm.auto import tqdm
import re
import equinox as eqx
import functools as ft
from itertools import product


#%%
@dataclass(frozen=True)
class ExpRecord:
    dataset: str
    d: int
    beta: float
    m: int
    kernel: str
    q0: Float[Array, "q d"]
    memories: Float[Array, "N d"]
    E0: Float[Array, "q"]
    E0_kernel: Float[Array, "q"]
    Eguess: Float[Array, "q"]
    gradEguess: Float[Array, "q d"]
    gradE0: Float[Array, "N d"]
    gradE0_kernel: Float[Array, "N d"]
    Emem: Float[Array, "N"] 
    Emem_kernel: Float[Array, "N"]
    Enear: Float[Array, "N"]
    gradEnear: Float[Array, "N d"]
    Enear_kernel: Float[Array, "N"]
    gradEnear_kernel: Float[Array, "N d"]
    gradEmem: Float[Array, "N d"]
    gradEmem_kernel: Float[Array, "N d"]
    E0_abserr: Float[Array, "q"]
    gradE0_abserr: Float[Array, "q d"]
    Emem_abserr: Float[Array, "N"]
    gradEmem_abserr: Float[Array, "N d"]
    Enear_abserr: Float[Array, "N"]
    gradEnear_abserr: Float[Array, "N d"]
    Eguess_abserr: Float[Array, "q"]
    gradEguess_abserr: Float[Array, "q d"]

    # Optional retrieval results
    did_retrieval: bool = False
    depth: int = None
    alpha: float = None
    depth_kernel: int = None
    alpha_kernel: float = None

    q0diff_hamming: Optional[Float[Array, "q"]] = None
    qneardiff_hamming: Optional[Float[Array, "q"]] = None
    memdiff_hamming: Optional[Float[Array, "N"]] = None
    
    ## Can optionally save the energies right now
    # Edescent: Optional[Float[Array, "q T"]] = None
    # Edescent_kernel: Optional[Float[Array, "q T"]] = None

    ## Can optionally store all the fixed points for everything
    # q0star: Optional[Float[Array, "q d"]] = None
    # q0star_kernel: Optional[Float[Array, "q d"]] = None
    # qnearstar: Optional[Float[Array, "q d"]] = None
    # qnearstar_kernel: Optional[Float[Array, "q d"]] = None
    # memstar: Optional[Float[Array, "N d"]] = None
    # memstar_kernel: Optional[Float[Array, "N d"]] = None

    def to_dict(self): return self.__dict__
    def to_series(self): return pd.Series(self.to_dict())

@dataclass(frozen=True)
class Args:
    """Compare the kernelized DAM energy, similarity, and updates to the standard DAM"""
    outdir: Optional[Union[Path,str]] = None # Where to save the results
    dataset: Optional[str] = None # Which dataset to use.
    kernel: str = 'SinCosL2DAM' # Which kernel to use
    seed: int = 0 # Random seed for all experiments
    dims: tuple[int, ...] = (30, 100, 500)
    nmemories: int = 500 # Number of memories to store, `K`
    ms: tuple[int, ...] = (1_000, 40_000, 80_000, 200_000) # Number of basis functions to test
    betas: tuple[float, ...] = (5., 20., 30., 40., 50.) # Which beta values to test. Use 'opt' if you want to test the optimal beta
    device: str = "1" # Which device to use. Leave "" for cpu
    redo_existing_ckpts: bool = True # If true, re-run experiments for checkpoints that exist
    pct_near_occlude:float = 0.1 # Frac of bits to occlude in "near memories" exp.

    do_retrieval: bool = False # If true, do retrieval experiments
    alpha: float = 0.03 # Inference step size down energy
    depth: int = 1000 # Number of DAM updates to make
    alpha_kernel: Optional[float] = None # Stepsize down kernelized energy. Defaults to `alpha`
    depth_kernel: Optional[float] = None # Num iterations down kernelized energy. Defaults to `depth`

    @property
    def nqueries(self):
        return self.nmemories

    def get_outdir(self):
        if self.outdir is None:
            outdir = pf.RESULTS / f"QUANT1b_near={self.pct_near_occlude}_retreival={self.do_retrieval}" 
        else:
            outdir = Path(self.outdir)
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        return outdir

    def get_ckptdir(self):
        ckptdir = self.get_outdir() / "ckpts"
        ckptdir.mkdir(exist_ok=True, parents=True)
        return ckptdir

    def get_device(self):
        return "" if self.device.lower() in ["", "cpu"] else self.device

    def get_alpha_kernel(self):
        return self.alpha if self.alpha_kernel is None else self.alpha_kernel
    
    def get_depth_kernel(self):
        return self.depth if self.depth_kernel is None else self.depth_kernel
    

def encode_key(key: jnp.ndarray) -> str: return jnp.array_str(key)
def decode_key(key: str) -> jnp.ndarray: 
    return jnp.fromstring(key[1:-1], sep=" ", dtype=jnp.uint32)
    
def main( args: Args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.get_device()
    
    rng = jr.PRNGKey(args.seed)

    dims = sorted(args.dims)
    ms = sorted(args.ms)

    *dkeys, rng = jr.split(rng, len(args.dims) + 1)
    dkeys = {k: v for k,v in zip(args.dims, dkeys)}

    *mkeys, rng = jr.split(rng, len(args.ms) + 1)
    mkeys = {k: v for k,v in zip(args.ms, mkeys)}

    existing_ckpts = list(args.get_ckptdir().glob("*.pkl"))

    if len(existing_ckpts) > 0:
        df_existing = pd.concat([pd.read_pickle(ckpt) for ckpt in existing_ckpts], ignore_index=True)
        print(f"Found {len(df_existing)} checkpoints")
        existing_dmb_combos = set([(d, m, beta, args.do_retrieval) for d, m, beta in zip(df_existing['d'].values, df_existing['m'].values, df_existing['beta'].values)])
    else:
        existing_dmb_combos = set([])
    
    istep = 0
    @ft.lru_cache(maxsize=2)
    def get_kdam(key: str, d, m, beta):
        key = decode_key(key)
        KernelClass = KernelOpts[args.kernel]
        kdam = KernelClass(key, d=d, m=m, beta=beta)
        return kdam

    @ft.lru_cache(maxsize=3)
    def get_patterns(key:str, d):
        # Where `key` is made by `jnp.array_str(key)``
        key = decode_key(key)
        kq, km, rng = jr.split(key, 3)
        memories = (jr.uniform(km, (10*args.nmemories, d)) > 0.5) * (1 / np.sqrt(d))
        memories = jnp.array(np.unique(memories, axis=0))[:args.nmemories]

        while True:
            # print(f"Are memories = {args.nmemories}? :: ", memories.shape[0])
            if memories.shape[0] < args.nmemories:
                km, rng = jr.split(rng)
                new_mems = (jr.uniform(km, (args.nmemories, d)) > 0.5) * (1 / np.sqrt(d))
                memories = np.concatenate([memories, new_mems], axis=0)
                memories = np.unique(memories, axis=0)
            else:
                break

        q0 = (jr.uniform(kq, (10*args.nqueries, d)) > 0.5) * (1 / np.sqrt(d))
        q0 = jnp.array(np.unique(q0, axis=0))[:args.nqueries]
        while True:
            # print(f"Are queries = {args.nqueries}? :: ", q0.shape[0])
            if q0.shape[0] < args.nqueries:
                kq, rng = jr.split(rng)
                new_q0 = (jr.uniform(kq, (args.nqueries, d)) > 0.5) * (1 / np.sqrt(d))
                q0 = np.concatenate([q0, new_q0], axis=0)
                q0 = np.unique(q0, axis=0)
            else:
                break

        n_occlude = int(args.pct_near_occlude * d)

        def occlude_pattern(key, pat, n_occlude):
            pat = pat * np.sqrt(d)
            occlude_idxs = jr.choice(key, d, (n_occlude,), replace=False)
            pat = pat.at[occlude_idxs].set((pat[occlude_idxs] + 1) % 2)
            return pat / np.sqrt(d)

        *keys, rng = jr.split(rng, memories.shape[0] + 1)
        keys = jnp.stack(keys)
        qnear = jax.vmap(lambda key, pat: occlude_pattern(key, pat, n_occlude=n_occlude), in_axes=(0, 0))(keys, memories)

        return memories, qnear, q0

    combos = list(product(dims[::-1], ms[::-1], args.betas))
    with tqdm(total=len(combos)) as pbar:
        for d, m, beta in combos:
            desc = f"Starting d={d}, m={m}, beta={beta}"
            pbar.update(1)
            if (d, m, beta, args.do_retrieval) in existing_dmb_combos: 
                continue
            
            pbar.set_description(desc)

            # Fix keys being changed every time?
            key = dkeys[d]
            memories, qnear, q0 = get_patterns(encode_key(key), d)

            # amkey, rng = jr.split(rng)
            amkey = mkeys[m]
            kdam = get_kdam(encode_key(amkey), d=d, m=m, beta=1.)

            if beta == "opt":
                kdam = kdam.condition_beta_on_memories(memories, check_hessian=False, atol=1e-6)
                print("Beta was 'opt': Smallest beta for well conditioned energy: ", kdam.beta)
                beta = f"opt_{kdam.beta}"
            else:
                kdam = eqx.tree_at(lambda model: model.beta, kdam, beta)

            T = kdam.kernelize_memories(memories)

            # Energy comparison
            E0, gradE0 = jax.vmap(jax.value_and_grad(kdam.energy), in_axes=(0, None))(q0, memories)
            Emem, gradEmem = jax.vmap(jax.value_and_grad(kdam.energy), in_axes=(0, None))(memories, memories)
            Enear, gradEnear = jax.vmap(jax.value_and_grad(kdam.energy), in_axes=(0, None))(qnear, memories)

            # E0_kernel, gradE0_kernel = logs['energies'][:,0], logs['grads'][:, 0]
            E0_kernel, gradE0_kernel = jax.vmap(jax.value_and_grad(kdam.kernel_energy), in_axes=(0, None))(q0, T)
            Emem_kernel, gradEmem_kernel = jax.vmap(jax.value_and_grad(kdam.kernel_energy), in_axes=(0, None))(memories, T)
            Enear_kernel, gradEnear_kernel = jax.vmap(jax.value_and_grad(kdam.kernel_energy), in_axes=(0, None))(qnear, T)

            if args.do_retrieval:
                q0star, _ = kdam.vrecall(q0, memories, alpha=args.alpha, depth=args.depth, return_grads=False)
                q0star = binarize_data(q0star)
                q0star_kernel, _ = kdam.vkernel_recall(q0, T, alpha=args.get_alpha_kernel(), depth=args.get_depth_kernel(), return_grads=False)
                q0star_kernel = binarize_data(q0star_kernel)

                qnearstar, _ = kdam.vrecall(qnear, memories, alpha=args.alpha, depth=args.depth, return_grads=False)
                qnearstar = binarize_data(qnearstar)
                qnearstar_kernel, _ = kdam.vkernel_recall(qnear, T, alpha=args.get_alpha_kernel(), depth=args.get_depth_kernel(), return_grads=False)
                qnearstar_kernel = binarize_data(qnearstar_kernel)

                memstar, _ = kdam.vrecall(memories, memories, alpha=args.alpha, depth=args.depth, return_grads=False)
                memstar = binarize_data(memstar)
                memstar_kernel, _ = kdam.vkernel_recall(memories, T, alpha=args.get_alpha_kernel(), depth=args.get_depth_kernel(), return_grads=False)
                memstar_kernel = binarize_data(memstar_kernel)

                # Normalized hamming distance
                q0diff_hamming = jnp.sum(jnp.abs(q0star - q0star_kernel), axis=-1) / jnp.sqrt(d)
                qneardiff_hamming = jnp.sum(jnp.abs(qnearstar - qnearstar_kernel), axis=-1) / jnp.sqrt(d)
                memdiff_hamming = jnp.sum(jnp.abs(memstar - memstar_kernel), axis=-1) / jnp.sqrt(d)

            else:
                q0diff_hamming = None
                qneardiff_hamming = None
                memdiff_hamming = None
                
            istep += 1
            key, rng = jr.split(rng)
            qrand = (jr.uniform(key, (args.nmemories, d)) > 0.5) * (1 / np.sqrt(d))
            Eguess, gradEguess = jax.vmap(jax.value_and_grad(kdam.kernel_energy), in_axes=(0, None))(qrand, T)
            
            record = ExpRecord(
                dataset="random",
                d=d,
                beta=beta,
                m=m,
                kernel=args.kernel,
                q0=q0,
                memories=memories,
                Eguess=Eguess,
                gradEguess=gradEguess,
                E0=E0,
                E0_kernel=E0_kernel,
                gradE0=gradE0,
                gradE0_kernel=gradE0_kernel,
                Emem=Emem,
                Emem_kernel=Emem_kernel,
                gradEmem=gradEmem,
                gradEmem_kernel=gradEmem_kernel,
                Enear=Enear,
                gradEnear=gradEnear,
                Enear_kernel=Enear_kernel,
                gradEnear_kernel=gradEnear_kernel,
                E0_abserr=jnp.abs(E0 - E0_kernel),
                Eguess_abserr=jnp.abs(Emem - Eguess),
                gradEguess_abserr=jnp.abs(gradEmem - gradEguess),
                gradE0_abserr=jnp.abs(gradE0 - gradE0_kernel),
                Emem_abserr=jnp.abs(Emem - Emem_kernel),
                gradEmem_abserr=jnp.abs(gradEmem - gradEmem_kernel),
                Enear_abserr=jnp.abs(Enear - Enear_kernel),
                gradEnear_abserr=jnp.abs(gradEnear - gradEnear_kernel),

                # Retrieval results
                did_retrieval=args.do_retrieval,
                depth=args.depth,
                alpha=args.alpha,
                depth_kernel=args.get_depth_kernel(),
                alpha_kernel=args.get_alpha_kernel(),

                q0diff_hamming = q0diff_hamming,
                qneardiff_hamming = qneardiff_hamming,
                memdiff_hamming = memdiff_hamming
            )
            fname = f"{istep:03d}--QUANT1b_retrieve={args.do_retrieval}_seed={args.seed}_m={m}_d={d}_beta={beta}_nmems={memories.shape[0]}_nqueries={q0.shape[0]}_kernel={args.kernel}.pkl"
            outf = args.get_ckptdir() / fname
            df = pd.DataFrame([record.to_dict()])
            df.to_pickle(outf)

#%%
if __name__ == "__main__":
    args = tyro.cli(Args)

    print("DEVICE: ", args.get_device())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.get_device()

    print(f"Saving results to {args.get_outdir()} and checkpoints to {args.get_ckptdir()}")
    main(args)
        
# %%
