#%%
"""
Kernelizing DAMs with various choices for basis functions.
"""
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import *
from jaxtyping import Float, Array, PyTree, jaxtyped, UInt, Bool
import equinox as eqx
import jax.random as jr
from fastcore.meta import delegates
from initializations import orthogonal_gaussian
import numpy as np
from tools import is_psd
import functools as ft

SIM_REGISTRY = {}

def register(cls):
    SIM_REGISTRY[cls.__name__] = cls
    return cls

# =============================================================================
# Define basis functions
# =============================================================================
def sin_cos_phi(x: Float[Array, "... d"], S:Float[Array, "m d"], b: Float[Array, "m"], beta: float, add_bias:bool) -> Float[Array, "... 2m"]:
    """The reigning champion, a basis function that uses sin and cos to encode the input into the feature space. """
    m = S.shape[0]
    h = jnp.sqrt(beta) * (x @ S.T)
    if add_bias:
        h = h + b
    return 1 / jnp.sqrt(m) * jnp.concatenate( [ jnp.cos(h), jnp.sin(h)], axis=-1)

def cos_phi(x: Float[Array, "... d"], S: Float[Array, "m d"], b: Float[Array, "m"], beta: float, add_bias:bool) -> Float[Array, "... m"]:
    """A basis function that uses only cos to encode the input into the feature space."""
    m = S.shape[0]
    h = jnp.sqrt(beta) * (x @ S.T)
    if add_bias:
        h = h + b
    return jnp.sqrt(2. / m) * jnp.cos(h)

def exp_phi(x: Float[Array, "... d"], S: Float[Array, "m d"], beta: float) -> Float[Array, "... m"]:
    """This is the basis function favored by Choromanski in his 'Performers' paper"""
    m = S.shape[0]
    h = jnp.sqrt(beta) * (x @ S.T)
    h = h - beta * jnp.sum(x ** 2, axis=-1)[..., None]
    return jnp.sqrt(1./m)*jnp.exp(h)

def exp_exp_phi(x: Float[Array, "... d"], S: Float[Array, "m d"], beta: float) -> Float[Array, "... 2m"]:
    """A theoretical improvement on the exp_phi basis function, but it's currently not working better..."""
    # DUPLICATED
    m = S.shape[0]
    h = jnp.sqrt(beta) * x @ S.T 
    cc = beta * jnp.sum(x ** 2, axis=-1)[..., None]
    z = jnp.concatenate([jnp.exp(h - cc), jnp.exp(-h - cc)], axis=-1)
    return 1 / jnp.sqrt(2*m) * z


# =============================================================================
# Defining the KernelizeableDAM
# =============================================================================
class KernelizableAM(ABC):
    """Defines the interface and basic methods for all KernelizableDAMs"""
    def kernel_sim(
        self, x: Float[Array, "d"], y: Float[Array, "d"]
    ) -> Float[Array, ""]:
        """Compute the energy of `x` against a set of `T` stored in basis function space."""
        return self.phi(x) @ self.phi(y)

    def kernelize_memories_chunk(self, memories: Float[Array, "n d"], chunk_size:int=20_000) -> Float[Array, "m"]:
        """Sometimes kernelizing the memories requires too much memory (i.e., when `m` features, `n` memories, and `d` dimensions are large)
        
        We can batch create the memory vector `T`"""
        n = memories.shape[0]
        nchunks = (n // chunk_size) + 1  #if n % chunk_size == 0 else (n // chunk_size) + 1
        Ts = jnp.zeros((nchunks, self.Tdim))

        for i in range(nchunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n)
            Ts = Ts.at[i].set(self.phi(memories[start:end]).sum(0))
        return Ts.sum(0)

    def kernelize_memories_basic(self, memories: Float[Array, "n d"], **kwargs) -> Float[Array, "m"]:
        """
        Naive implementation that BLOWS UP with many memories `n`, since it creates the entire memory matrix from scratch
        """
        return self.phi(memories).sum(0)

    def kernelize_memories(self, 
                           memories: Float[Array, "n d"], # Memories to encode in the kernel
                           optimize_at: int = 30_000, # If the number of memories is less than this, use the basic kernelize_memories function
                           **kwargs # Passed to the specific kernelize_memories function
                           ) -> Float[Array, "m"]:
        """Given a memory matrix, compute the kernelized memory matrix `T` by summing the feature representations of every memory"""
        if memories.shape[0] < optimize_at:
            return self.kernelize_memories_basic(memories, **kwargs)
        return self.kernelize_memories_chunk(memories, **kwargs)

    def recall( self, q: Float[Array, "d"], memories: Float[Array, "n d"], depth: int=1000, alpha: float = 0.1, return_grads=False, clamp_idxs: Optional[Bool[Array, "d"]]=None) -> Float[Array, "d"]: 
        """Using the normal similarity function, run energy dynamics"""
        dEdxf = jax.jit(jax.value_and_grad(self.energy))
        # dEdxf = jax.value_and_grad(self.energy)
        logs = {}
        def step(x, i):
            E, dEdx = dEdxf(x, memories)
            if clamp_idxs is not None:
                dEdx = jnp.where(clamp_idxs, 0, dEdx)
            x = x - alpha * dEdx
            aux = (E, dEdx) if return_grads else (E,)
            return x, aux
        x, aux = jax.lax.scan(step, q, jnp.arange(depth))
        logs['energies'] = aux[0]
        if return_grads:
            logs['grads'] = aux[1]
        return x, logs

    @delegates(recall)
    def vrecall( self, q: Float[Array, "b d"], memories: Float[Array, "n d"], **kwargs) -> Float[Array, "d"]: 
        """Run energy dynamics with simple gradient descent on a batch of queries """
        recallf = ft.partial(self.recall, **kwargs)
        return jax.vmap(recallf, in_axes=(0, None))(q, memories)

    def kernel_recall( self, q: Float[Array, "d"], T: Float[Array, "m"], depth: int=1000, alpha: float = 0.1, return_grads=False, clamp_idxs: Optional[Bool[Array, "d"]]=None) -> Float[Array, "d"]: 
        """Using the kernelized similarity function, run energy dynamics"""
        dEdxf = jax.jit(jax.value_and_grad(self.kernel_energy))
        logs = {}
        @jax.jit
        def step(x, i):
            E, dEdx = dEdxf(x, T)
            if clamp_idxs is not None:
                dEdx = jnp.where(clamp_idxs, 0, dEdx)
            x = x - alpha * dEdx
            aux = (E, dEdx) if return_grads else (E,)
            return x, aux
        x, aux = jax.lax.scan(step, q, jnp.arange(depth))
        logs['energies'] = aux[0]
        if return_grads:
            logs['grads'] = aux[1]
        return x, logs

    @delegates(kernel_recall)
    def vkernel_recall( self, q: Float[Array, "d"], T: Float[Array, "m"], **kwargs) -> Float[Array, "d"]: 
        kernel_recallf = ft.partial(self.kernel_recall, **kwargs)
        return jax.vmap(kernel_recallf, in_axes=(0, None))(q, T)


    @abstractmethod
    def sim(self, x: Float[Array, "d"], y: Float[Array, "d"]) -> Float[Array, ""]: ...

    @abstractmethod
    def phi(self, x: Float[Array, "... d"]) -> Float[Array, "... m"]: ...

    @abstractmethod
    def stochastic_phi(self, x: Float[Array, "... d"], midxs: UInt[Array, "y"]) -> Float[Array, "... y"]: ...

    @abstractmethod
    def energy( self, x: Float[Array, "d"], memories: Float[Array, "n d"]) -> Float[Array, ""]: ...

    @abstractmethod
    def kernel_energy( self, x: Float[Array, "d"], T: Float[Array, "m"], **kwargs) -> Float[Array, ""]: ...

    @abstractmethod
    def stochastic_kernel_energy( self, x: Float[Array, "d"], T: Float[Array, "m"], midxs: UInt[Array, "y"], **kwargs) -> Float[Array, ""]: ...


    def is_well_conditioned(self, 
                            memories: Float[Array, "n d"], 
                            other_points: Float[Array, "s d"] = None, # If none, compare conditioning on the memories themselves
                            atol=1e-7, # Gradient must be within this absolute tolerance
                            rtol=1e-5, # Gradient must be within this relative tolerance
                            check_hessian=False # Whether to check hessian PSD about memories
                            ):
        """Check if the standard energy function is "well conditioned" about a set of memories -- that is:
        
        Does each memory live at a local energy minimum? This is true if the following two conditions hold:
            1: the gradient around that memory is 0 
            2: the hessian is positive semi-definite (optional check, more memory intensive)
        """
        points = memories if other_points is None else other_points
        dEs = jax.vmap(jax.grad(self.energy), in_axes=(0, None))(points, memories)

        if jnp.allclose(dEs, 0, atol=atol, rtol=rtol):
            if check_hessian:
                print("Grads near 0! Checking hessian")
                ddEs = jax.vmap(jax.hessian(self.energy), in_axes=(0, None))(points, memories)
                return jnp.all(jax.vmap(is_psd)(ddEs))
            else:
                return True
        else:
            return False

    @delegates(is_well_conditioned)
    def condition_beta_on_memories(self, 
                                   memories:Float[Array, "n d"], # Which memories to condition on
                                   beta_min=1e-4, # Minimum guess for beta
                                   beta_max=600., 
                                   max_steps=40, # Maximum number of steps to run the binary search for
                                   precision=1e-3, # Stop the search when max-min < this precision
                                   **kwargs # Passed to the is_well_conditioned function
                                   ):
        @dataclass
        class BinarySearchState:
            min: float
            max: float
            def __post_init__(self): assert self.min < self.max

            @property
            def now(self): return (self.max + self.min) / 2

        beta = BinarySearchState(beta_min, beta_max)
        for i in range(max_steps):
            kdam = eqx.tree_at(lambda model: model.beta, self, beta.now)
            if kdam.is_well_conditioned(memories, **kwargs): 
                beta.max = beta.now
                if beta.max - beta.min < precision: break
            else: beta.min = beta.now

        print(f"Found beta={beta.now} in {i} steps")
        return kdam

@register
class CosL2DAM(eqx.Module, KernelizableAM):
    beta: float
    m: int
    d: int
    S: jax.Array
    b: jax.Array
    add_bias: bool
    Tdim: int

    def __init__(self, key, d, m, beta, add_bias=True, orthogonal_init=False):
        k1, k2 = jr.split(key)
        if orthogonal_init:
            self.S = orthogonal_gaussian(k1, m, d)
        else:
            self.S = jr.normal(k1, (m, d))
        self.b = jr.uniform(k2, shape=(m,), minval=0.0, maxval=2.0 * jnp.pi)
        self.add_bias = add_bias
        self.beta = beta
        self.m = m
        self.Tdim = m
        self.d = d

    def sim(self, x: Float[Array, "d"], y: Float[Array, "d"]) -> Float[Array, ""]:
        """Compute the standard L2 similarity between two vectors."""
        return jnp.exp(-self.beta / 2 * ((x - y) ** 2).sum())

    def phi(self, x: Float[Array, "... d"]) -> Float[Array, "... m"]:
        """Compute the basis function """
        return cos_phi(x, self.S, self.b, self.beta, self.add_bias)

    def stochastic_phi(self, x: Float[Array, "... d"], midxs: UInt[Array, "y"]) -> Float[Array, "... m"]:
        """Compute the basis function """
        return cos_phi(x, self.S[midxs], self.b[midxs], self.beta, self.add_bias)

    def energy(
        self, x: Float[Array, "d"], memories: Float[Array, "n d"]
    ) -> Float[Array, ""]:
        """Compute the standard L2 energy"""
        return -(1 / self.beta) * jax.nn.logsumexp(
            -self.beta / 2 * ((x - memories) ** 2).sum(-1), axis=0
        )

    def kernel_energy(
        self, x: Float[Array, "d"], T: Float[Array, "m"], eps=1e-5
    ) -> Float[Array, ""]:
        """Compute the approximate kernelized energy"""
        h = self.phi(x) @ T
        h = jnp.clip(h,  a_min=eps)
        return -(1 / self.beta) * jnp.log(h)

    def stochastic_kernel_energy(
        self, x: Float[Array, "d"], T: Float[Array, "m"], midxs: UInt[Array, "y"], eps=1e-5
    ) -> Float[Array, ""]:
        """Compute the approximate kernelized energy"""
        mSubset = len(midxs)
        mTotal = self.m
        h = jnp.sqrt(mTotal / mSubset) * self.stochastic_phi(x, midxs) @ T[midxs]
        h = jnp.clip(h,  a_min=eps)
        return -(1 / self.beta) * jnp.log(h)

@register
class CosNoLogL2DAM(CosL2DAM):
    def energy( self, x: Float[Array, "d"], memories: Float[Array, "n d"]) -> Float[Array, ""]:
        return -(1 / self.beta) * jnp.sum(jnp.exp(-self.beta / 2 * ((x - memories) ** 2).sum(-1)), axis=0)

    def kernel_energy( self, x: Float[Array, "d"], T: Float[Array, "m"], **kwargs) -> Float[Array, ""]:
        return -(1/self.beta) * self.phi(x) @ T

    def stochastic_kernel_energy(
        self, x: Float[Array, "d"], T: Float[Array, "m"], midxs: UInt[Array, "y"], eps=1e-5
    ) -> Float[Array, ""]:
        """Compute the approximate kernelized energy"""
        mSubset = len(midxs)
        mTotal = self.m
        h = jnp.sqrt(mTotal / mSubset) * self.stochastic_phi(x, midxs) @ T[midxs]
        h = jnp.clip(h,  a_min=eps)
        return -(1 / self.beta) * h

@register
class SinCosL2DAM(CosL2DAM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Tdim = 2 * self.m
        
    def phi(self, x: Float[Array, "... d"]) -> Float[Array, "... 2m"]:
        out = sin_cos_phi(x, self.S, self.b, self.beta, self.add_bias)
        return out 

    def stochastic_phi(self, x: Float[Array, "... d"], midxs: UInt[Array, "y"]) -> Float[Array, "... 2m"]:
        """Compute the basis function """
        return sin_cos_phi(x, self.S[midxs], self.b[midxs], self.beta, self.add_bias)

    def stochastic_kernel_energy(
        self, x: Float[Array, "d"], T: Float[Array, "m"], midxs: UInt[Array, "y"], eps=1e-5
    ) -> Float[Array, ""]:
        """Compute the approximate kernelized energy"""
        mSubset = len(midxs)
        mTotal = self.m
        subT = jnp.concatenate([T[midxs], T[midxs + self.m]], axis=-1)
        h = jnp.sqrt(mTotal / mSubset) * self.stochastic_phi(x, midxs) @ subT
        h = jnp.clip(h,  a_min=eps)
        return -(1 / self.beta) * jnp.log(h)

@register
class SinCosNoLogL2DAM(CosNoLogL2DAM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Tdim = 2 * self.m

    def phi(self, x: Float[Array, "... d"]) -> Float[Array, "... 2m"]:
        return sin_cos_phi(x, self.S, self.b, self.beta, self.add_bias)

    def stochastic_phi(self, x: Float[Array, "... d"], midxs: UInt[Array, "y"]) -> Float[Array, "... 2m"]:
        return sin_cos_phi(x, self.S[midxs], self.b[midxs], self.beta, self.add_bias)

    def stochastic_kernel_energy(
        self, x: Float[Array, "d"], T: Float[Array, "m"], midxs: UInt[Array, "y"], eps=1e-5
    ) -> Float[Array, ""]:
        """Compute the approximate kernelized energy"""
        mSubset = len(midxs)
        mTotal = self.m
        subT = jnp.concatenate([T[midxs], T[midxs + self.m]], axis=-1)
        h = jnp.sqrt(mTotal / mSubset) * self.stochastic_phi(x, midxs) @ subT
        h = jnp.clip(h,  a_min=eps)
        return -(1 / self.beta) * jnp.log(h)

@register
class ExpL2DAM(CosL2DAM):
    def phi(self, x: Float[Array, "... d"]) -> Float[Array, "... m"]:
        return exp_phi(x, self.S, self.beta)

    def stochastic_phi(self, x: Float[Array, "... d"], midxs: UInt[Array, "y"]) -> Float[Array, "... m"]:
        return exp_phi(x, self.S[midxs], self.beta)

    def kernel_energy(
        self, x: Float[Array, "d"], T: Float[Array, "m"], eps=0.
    ) -> Float[Array, ""]:
        # Forces eps to 0 inherently!
        return super().kernel_energy(x, T, eps=0.)

    def stochastic_kernel_energy(
        self, x: Float[Array, "d"], T: Float[Array, "m"], midxs: UInt[Array, "y"], eps=0.
    ) -> Float[Array, ""]:
        # Forces eps to 0 inherently!
        return super().stochastic_kernel_energy(x, T, midxs, eps=0.)

@register
class ExpNoLogL2DAM(CosNoLogL2DAM):
    def phi(self, x: Float[Array, "... d"]) -> Float[Array, "... m"]:
        return exp_phi(x, self.S, self.beta)

    def stochastic_phi(self, x: Float[Array, "... d"], midxs: UInt[Array, "y"]) -> Float[Array, "... m"]:
        return exp_phi(x, self.S[midxs], self.beta)

    def kernel_energy(
        self, x: Float[Array, "d"], T: Float[Array, "m"], eps=0.
    ) -> Float[Array, ""]:
        return super().kernel_energy(x, T, eps=0.)

    def stochastic_kernel_energy(
        self, x: Float[Array, "d"], T: Float[Array, "m"], midxs: UInt[Array, "y"], eps=0.
    ) -> Float[Array, ""]:
        return super().stochastic_kernel_energy(x, T, midxs, eps=0.)

#%%
@register
class ExpExpL2DAM(CosL2DAM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Tdim = 2 * self.m

    def phi(self, x: Float[Array, "... d"]) -> Float[Array, "... 2m"]:
        return exp_exp_phi(x, self.S, self.beta)

    def stochastic_phi(self, x: Float[Array, "... d"], midxs: UInt[Array, "y"]) -> Float[Array, "... 2m"]:
        return exp_exp_phi(x, self.S[midxs], self.beta)

    def kernel_energy(
        self, x: Float[Array, "d"], T: Float[Array, "m"], eps=0.
    ) -> Float[Array, ""]:
        return super().kernel_energy(x, T, eps=0.)

    def stochastic_kernel_energy(
        self, x: Float[Array, "d"], T: Float[Array, "m"], midxs: UInt[Array, "y"], eps=0.
    ) -> Float[Array, ""]:
        return super().stochastic_kernel_energy(x, T, midxs, eps=0.)

@register
class ExpExpNoLogL2DAM(CosNoLogL2DAM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Tdim = 2 * self.m

    def phi(self, x: Float[Array, "... d"]) -> Float[Array, "... 2m"]:
        return exp_exp_phi(x, self.S, self.beta)

    def stochastic_phi(self, x: Float[Array, "... d"], midxs: UInt[Array, "y"]) -> Float[Array, "... 2m"]:
        return exp_exp_phi(x, self.S[midxs], self.beta)

    def kernel_energy(
        self, x: Float[Array, "d"], T: Float[Array, "m"], eps=0.
    ) -> Float[Array, ""]:
        return super().kernel_energy(x, T, eps=0.)

    def stochastic_kernel_energy(
        self, x: Float[Array, "d"], T: Float[Array, "m"], midxs: UInt[Array, "y"], eps=0.
    ) -> Float[Array, ""]:
        return super().stochastic_kernel_energy(x, T, midxs, eps=0.)

#%%
if __name__ == "__main__":
    rng = jr.PRNGKey(0)
    k1, k2, k3, k4, rng = jr.split(rng, 5)
    d = 100
    m = 10000
    n_memories = 3333
    beta = 40.0
    kdam = CosL2DAM(k1, d=d, m=m, beta=beta)

    x = (jr.uniform(k2, (d,)) > 0.5) / jnp.sqrt(d)
    y = (jr.uniform(k3, (d,)) > 0.5) / jnp.sqrt(d)

    print(kdam.sim(x, y))
    print(kdam.kernel_sim(x, y))

    memories = jr.uniform(k4, (n_memories, d)) > 0.5 / jnp.sqrt(d)

    print(kdam.energy(x, memories))
    T = kdam.kernelize_memories(memories)
    print(kdam.kernel_energy(x, T))

    kdam = SinCosL2DAM(k1, d=d, m=m, beta=beta)
    print(kdam.phi(jnp.stack([x,y], axis=0)).shape)

    kdam = ExpExpL2DAM(k1, d=d, m=m, beta=beta)
    print(kdam.phi(jnp.stack([x,y], axis=0)).shape)
# %%
