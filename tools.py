import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Array, PyTree, jaxtyped
from typing import Tuple
import numpy as np
from tqdm.auto import tqdm

def outerize(f):
    """Given a binary (2-arg) function `f`, compute the outer product if each arg has a batch dimension
    
    Usage:
    ```
        def f(x, y):
            return x @ y
        f(jnp.ones(5), jnp.ones(5)) # scalar

        fouter = outerize(f)
        fouter(jnp.ones((5, 3)), jnp.ones((5, 3))) # 5x5 matrix
    ```
    """
    return jax.vmap(jax.vmap(f, in_axes=(None, 0)), in_axes=(0, None))

def is_psd(X):
    L = jnp.linalg.cholesky(X)
    return jnp.isnan(L).sum() == 0

def init_queries_for_all_memories(
    rng, # Random seed
    memories: Float[Array, "n d"], # We want to init a query for each of these memories
    nq0=None,  # Guess number of queries needed to find all memories. If None, default to the number of memories
    r=1.3, # Ratio with which to increase our guess for the number of queries by
    ensure_unique=True, # If true, ensure all generated queries are unique
    ) -> Tuple[Float[Array, "q d"], Float[Array, "n"]]:
    """Brute force initialize queries for all memories by randomly guessing queries until each memory has a closest query.
    
    Relies on L2 distance. If any queries are duplicated, remove them
    """
    N, D = memories.shape
    _nq = nq0 or N
    i = 0

    pbar = tqdm()

    while True:
        i+=1
        pbar.update(1)
        pbar.set_description(f"Step {i}")
        kq, rng = jr.split(rng)
        queries = (jr.uniform(kq, (_nq, D)) > 0.5) * (1 / np.sqrt(D))
        l2dists = outerize(lambda x,y: jnp.linalg.norm(x-y, axis=-1))(queries, memories)
        closest_mem = jnp.argmin(l2dists, axis=-1)
        Nunique = len(jnp.unique(closest_mem))
        if Nunique < N:
            _nq = int(r * _nq)
        elif Nunique == N:
            print(f"Found {Nunique}={N} unique memories using nq={_nq}")
            break
    
    # ensure uniqueness
    if ensure_unique:
        Nog = queries.shape[0]
        queries = jnp.array(np.unique(queries, axis=0))
        l2dists = outerize(lambda x,y: jnp.linalg.norm(x-y, axis=-1))(queries, memories)
        closest_mem = jnp.argmin(l2dists, axis=-1)
        print(f"After checking uniqueness, reduced query size from {Nog} to {queries.shape[0]}")

    return queries, closest_mem

def binarize_data(x: Float[Array, "... d"]):
    D = x.shape[-1]
    midpoint = 0.5 / np.sqrt(D)
    return (x > midpoint).astype(np.float32) / np.sqrt(D)

def discretize(valid_vals:Float[Array, "k"], # The valid discrete values that `x` can take
                    x:Float[Array, "d"] # The vector we want to bin
                    ) -> Float[Array, "d"]:
    """Discretize data `x` into `k` bins as defined by `valid_vals`.
    Each value in x is pulled to the closest valid value in `valid_vals`.
    
    Example (binary):
    > vals = np.array([0,1])
    > x = np.array([0.1, 0.2, -50., -0.0001, 0, 1.5, 1.1, 0.5])
    > xout = discretize_data(vals, x) # array([0, 0, 0, 0, 0, 1, 1, 1])

    """
    val_midpoints = (valid_vals[1:] + valid_vals[:-1]) / 2
    binvals = np.concatenate([[-np.inf], val_midpoints, [np.inf]])
    return np.digitize(x, binvals) - 1