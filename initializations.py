"""
Work in progress. Can we improve the approximation of the kernel if we use orthogonal Gaussian random features instead of straight random features?
"""
#%%
from typing import *
import jax
import jax.numpy as jnp
import jax.random as jr


# JAX implementation with Gram-Schmidt normalization
def orthogonal_gaussian(key: jax.Array, m:int, d:int):
    def orthogonal_square(key: jax.Array):
        q, _ = jnp.linalg.qr(jr.normal(key, shape=(d,d)))
        return q.T

    num_squares = m // d
    *keys, rng = jr.split(key, num_squares + 1)
    blocks = [orthogonal_square(k) for k in keys]
    remainder = m - d * num_squares
    if remainder:
        key, rng = jr.split(rng)
        blocks.append(orthogonal_square(key)[:remainder])
    matrix = jnp.vstack(blocks)
    matrix /= jnp.sqrt(num_squares + remainder / d)
    # matrix = jnp.diag(jnp.sqrt(d) * jnp.ones(m)) @ matrix
    return matrix

# def orthogonal_gaussian(key, m, d):
#     H = jr.normal(key, shape=(m, d))
#     u, s, vh = jnp.linalg.svd(H, full_matrices=False)
#     mat = u @ vh
#     return mat

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    d = 100
    m = 1000
    key = jr.PRNGKey(0)
    mat = orthogonal_gaussian(key, m, d)
    plt.imshow(mat.T @ mat)
    print(mat.shape)

    # import numpy as np
    # # generate IID Gaussian random features
    # def orthogonal_gaussian(m, d):
    #     def orthogonal_square():
    #         # create orthogonal square matrix using Gram-Schmidt
    #         q, _ = np.linalg.qr(np.random.normal(size=(d, d)))
    #         return q.T

    #     num_squares = int(m / d)
    #     blocks = [orthogonal_square() for _ in range(num_squares)]

    #     remainder = m - d * num_squares
    #     if remainder:
    #         blocks.append(orthogonal_square()[:remainder])

    #     matrix = np.vstack(blocks)
    #     matrix /= np.sqrt(num_squares + remainder / d)
    #     # matrix = np.diag(np.sqrt(d) * np.ones(m)) @ matrix

    #     return matrix

    # mat2 = orthogonal_gaussian(m, d)
    # plt.imshow(mat.T @ mat)
    # mat2.shape
    
# %%
## Original implementation
# import numpy as np

# # generate IID Gaussian random features
# def iid_gaussian(m, d):
#     return np.random.normal(size=(m, d))
# def orthogonal_gaussian(m, d):
#     def orthogonal_square():
#         # create orthogonal square matrix using Gram-Schmidt
#         q, _ = np.linalg.qr(iid_gaussian(d, d))
#         return q.T

#     num_squares = int(m / d)
#     blocks = [orthogonal_square() for _ in range(num_squares)]

#     remainder = m - d * num_squares
#     if remainder:
#         blocks.append(orthogonal_square()[:remainder])

#     matrix = np.vstack(blocks)
#     matrix /= np.sqrt(num_squares + remainder / d)
#     # matrix = np.diag(np.sqrt(d) * np.ones(m)) @ matrix

#     return matrix

