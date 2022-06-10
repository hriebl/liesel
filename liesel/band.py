from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def bandwidth(full):
    """
    Returns the bandwidth p of `full` as an integer.

    ## Parameters

    - `full`: A full (n x n) matrix.
    """


@partial(jax.jit, static_argnames="bandwidth")
def to_band(full, bandwidth, lower=True):
    """
    ## Parameters

    - `full`: A full (n x n) matrix.
    - `bandwidth`: The bandwidth p.
    - `lower`: Whether to return the (p x n) band in lower-diagonal ordered form.
    """


# no @jax.jit
def permute(full, lower=True):
    """
    Permutes `full` with the reverse Cuthill-McKee algorithm and returns the result
    as a (p x n) band.

    ## Parameters

    - `full`: A full (n x n) matrix.
    - `lower`: Whether to return the (p x n) band in lower-diagonal ordered form.
    """

    # use scipy.sparse.csgraph.reverse_cuthill_mckee here


@jax.jit
def to_symm_full(band, lower=True):
    """
    Returns the symmetric full (n x n) matrix represented by `band`.

    ## Parameters

    - `band`: A (p x n) band.
    - `lower`: Whether `band` is in lower-diagonal ordered form.
    """


@jax.jit
def to_ltri_full(band, lower=True):
    """
    Returns the lower triangular full (n x n) matrix represented by `band`.

    ## Parameters

    - `band`: A (p x n) band.
    - `lower`: Whether `band` is in lower-diagonal ordered form.
    """


@jax.jit
def to_utri_full(band, lower=True):
    """
    Returns the upper triangular full (n x n) matrix represented by `band`.

    ## Parameters

    - `band`: A (p x n) band.
    - `lower`: Whether `band` is in lower-diagonal ordered form.
    """


@jax.jit
def to_lower(band):
    """
    ## Parameters

    - `band`: A (p x n) band in upper-diagonal ordered form.
    """


@jax.jit
def to_upper(band):
    """
    ## Parameters

    - `band`: A (p x n) band in lower-diagonal ordered form.
    """

    # TODO: make this jittable
    band = np.array([np.roll(row, i) for i, row in enumerate(band)])
    return np.flip(band, 0)


@jax.jit
def cholesky(band, lower=True):
    """
    Returns the Cholesky decomposition of `band` as a (p x n) band in lower-diagonal
    ordered form.

    ## Parameters

    - `band`: A (p x n) band.
    - `lower`: Whether `band` is in lower-diagonal ordered form.
    """

    band = jax.lax.cond(lower, lambda band: band, to_lower)

    p = band.shape[0] - 1
    n = band.shape[1]

    L = jnp.zeros_like(band)

    def outer(j, L):
        v = band[:, j]

        def inner(k, v):
            def add(i, v):
                return v.at[i - j + k].add(-(L[i, k] * L[j - k, k]))

            return jax.lax.fori_loop(j - k, p + 1, add, v)

        start = jnp.clip(j - p, a_min=0, a_max=None)
        v = jax.lax.fori_loop(start, j, inner, v)

        return L.at[:, j].set(v / jnp.sqrt(v[0]))

    return jax.lax.fori_loop(0, n, outer, L)


@jax.jit
def backward_sub(band, rhs, lower=True):
    """
    Solves `L @ x = b`, where `L = to_ltri_full(band, lower)` and `b = rhs`.

    ## Parameters

    - `band`: A (p x n) band.
    - `rhs`: The right-hand side of the system of linear equations.
    - `lower`: Whether `band` is in lower-diagonal ordered form.
    """

    band = jax.lax.cond(lower, to_upper, lambda band: band)

    # TODO: make this jittable

    p = band.shape[0] - 1
    n = band.shape[1]

    x = np.zeros(n + p)

    for i in range(n):
        x[i + p] = (rhs[i] - x[i : (i + p + 1)] @ band[:, i]) / band[p, i]

    return x[p:]


@jax.jit
def forward_sub(band, rhs, lower=True):
    """
    Solves `U @ x = b`, where `U = to_utri_full(band, lower)` and `b = rhs`.

    ## Parameters

    - `band`: A (p x n) band.
    - `rhs`: The right-hand side of the system of linear equations.
    - `lower`: Whether `band` is in lower-diagonal ordered form.
    """

    band = jax.lax.cond(lower, lambda band: band, to_lower)

    # TODO: make this jittable

    p = band.shape[0] - 1
    n = band.shape[1]

    x = np.zeros(n + p)

    for i in reversed(range(n)):
        x[i] = (rhs[i] - x[i : (i + p + 1)] @ band[:, i]) / band[0, i]

    return x[:n]
