from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps


# @jax.jit not useful, f is compiled inside scan anyway, this makes it possible to return int.
def bandwidth(full) -> int:
    """
    Returns the bandwidth p of `full` as an integer.

    ## Parameters

    - `full`: A full (n x n) matrix.
    """

    def f(j, row):  # Iterate over Row
        def g(i, val):  # Iterate over value in Row
            return i + 1, jax.lax.cond(val != 0, jnp.abs, lambda _: -1, i - j)

        _, L = jax.lax.scan(g, 0, row)
        return j + 1, L

    _, b = jax.lax.scan(f, 0, full)
    return int(jnp.max(b))


@partial(jax.jit, static_argnames="bw")
def to_band(full, bw, lower=True):
    """
    ## Parameters

    - `full`: A full (n x n) matrix.
    - `bw`: The bandwidth p.
    - `lower`: Whether to return the (p x n) band in lower-diagonal ordered form.
    """
    n = full.shape[0]
    bw += 1

    # We only access the upper triangular. If we get a Matrix in lower-triangular (L) form, we convert.
    full = jnp.where(jnp.array_equal(full, jnp.tril(full)), full.T, full)

    # TODO: lax.fori_loop here?

    def lo():
        Ab = jnp.zeros((bw, n))
        for k in range(bw):
            Ab = Ab.at[k, :(n - k)].set(jnp.diag(full, k))
        return Ab

    def up():
        Ab = jnp.zeros((bw, n))
        for k in range(bw):
            Ab = Ab.at[k, k:].set(jnp.diag(full, k))
        return jnp.flipud(Ab)

    return jax.lax.cond(lower, lo, up)


# no @jax.jit
def permute(full, lower=True):
    """
    Permutes `full` with the reverse Cuthill-McKee algorithm and returns the result
    as a (p x n) band.

    ## Parameters

    - `full`: A full, symmetric(!) (n x n) matrix.
    - `lower`: Whether to return the (p x n) band in lower-diagonal ordered form.
    """

    if not np.allclose(full, full.T):
        raise ValueError("Argument `full` is not symmetric.")

    p = sps.csgraph.reverse_cuthill_mckee(sps.csr_matrix(full), symmetric_mode=True)
    P = np.vstack([np.eye(1, full.shape[0], k) for k in p])
    fullP = P @ full @ P.T

    # TODO: We need to return P to solve back?
    return to_band(fullP, bandwidth(fullP), lower)


@jax.jit
def to_symm_full(band, lower=True):
    """
    Returns the symmetric full (n x n) matrix represented by `band`.

    ## Parameters

    - `band`: A (p x n) band with bandwidth p+1.
    - `lower`: Whether `band` is in lower-diagonal ordered form.
    """
    a = to_utri_full(band, lower)
    return a + jnp.tril(a.T, k=-1)


@jax.jit
def to_ltri_full(band, lower=True):
    """
    Returns the lower triangular full (n x n) matrix represented by `band`.

    ## Parameters

    - `band`: A (p x n) band with bandwidth p+1.
    - `lower`: Whether `band` is in lower-diagonal ordered form.
    """
    return to_utri_full(band, lower).T


@jax.jit
def to_utri_full(band, lower=True):
    """
    Returns the upper triangular full (n x n) matrix represented by `band`.

    ## Parameters

    - `band`: A (p x n) band with bandwidth p+1.
    - `lower`: Whether `band` is in lower-diagonal ordered form.
    """
    p, n = band.shape
    p -= 1
    return jax.lax.cond(lower,
                        lambda: sum([jnp.diag(v[:n - k], k) for k, v in enumerate(band)]),  # lower
                        lambda: sum([jnp.diag(band[k][p - k:], p - k) for k in range(p + 1)]))  # upper


@jax.jit
def to_lower(band):
    """
    ## Parameters

    - `band`: A (p x n) band in upper-diagonal ordered form.
    """
    return jnp.flipud(jnp.array([jnp.roll(row, i - band.shape[0] + 1) for i, row in enumerate(band)]))


@jax.jit
def to_upper(band):
    """
    ## Parameters

    - `band`: A (p x n) band in lower-diagonal ordered form.
    """
    return jnp.flipud(jnp.array([jnp.roll(row, i) for i, row in enumerate(band)]))


@jax.jit
def cholesky(band, lower=True):
    """
    Returns the Cholesky decomposition of `band` as a (p x n) band in lower-diagonal
    ordered form.

    ## Parameters

    - `band`: A (p x n) band.
    - `lower`: Whether `band` is in lower-diagonal ordered form.
    """

    band = jax.lax.cond(lower, lambda x: x, to_lower, band)

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
def forward_sub(band, rhs, lower=True):
    """
    Solves `L @ x = b`, where `L = to_ltri_full(band, lower)` and `b = rhs`.

    ## Parameters

    - `band`: A (p x n) band.
    - `rhs`: The right-hand side of the system of linear equations.
    - `lower`: Whether `band` is in lower-diagonal ordered form.
    """

    band = jax.lax.cond(lower, to_upper, lambda x: x, band)

    p = band.shape[0] - 1
    n = band.shape[1]

    x = jnp.zeros(n + p)

    for i in range(n):
        x = x.at[i + p].set((rhs[i] - jnp.dot(x[i: (i + p + 1)], band[:, i])) / band[p, i])

    return x[p:]


@jax.jit
def backward_sub(band, rhs, lower=True):
    """
    Solves `U @ x = b`, where `U = to_utri_full(band, lower)` and `b = rhs`.

    ## Parameters

    - `band`: A (p x n) band.
    - `rhs`: The right-hand side of the system of linear equations.
    - `lower`: Whether `band` is in lower-diagonal ordered form.
    """

    band = jax.lax.cond(lower, lambda x: x, to_lower, band)

    p = band.shape[0] - 1
    n = band.shape[1]

    x = jnp.zeros(n + p)

    for i in reversed(range(n)):
        x = x.at[i].set((rhs[i] - jnp.dot(x[i: (i + p + 1)], band[:, i])) / band[0, i])

    return x[:n]