import numpy as np
import pytest
import scipy.linalg as linalg
import scipy.sparse as sps
from pytest import approx

from liesel import band

n = 5
p = 1


@pytest.fixture
def S():
    """Provides an example pos-definite Matrix."""
    S = sps.rand(n, n, density=0.8, random_state=42).toarray()
    S = S @ S.T  # Make S symmetric
    return np.triu(np.tril(S, p), -p) + np.eye(n) * n  # Make S banded & pos-def.


def test_bandwidth(S):
    assert band.bandwidth(np.eye(3)) == 0
    assert band.bandwidth(np.zeros((3, 3))) == 0
    assert band.bandwidth(S) == p


def test_to_band(S):
    assert band.to_band(np.eye(5), 0) == approx(np.ones((1, 5)))
    assert band.to_band(S, p).shape == (p + 1, n)
    assert band.to_band(np.tril(S), p) == approx(band.to_band(np.triu(S), p))


def test_to_band_conversions(S):
    Sb = band.to_band(S, p)
    assert band.to_symm_full(Sb) == approx(S)
    assert band.to_ltri_full(Sb) == approx(np.tril(S))
    assert band.to_utri_full(Sb) == approx(np.triu(S))

    Sb = band.to_band(S, p, False)
    assert band.to_symm_full(Sb, False) == approx(S)
    assert band.to_ltri_full(Sb, False) == approx(np.tril(S))
    assert band.to_utri_full(Sb, False) == approx(np.triu(S))


def test_band_conversions(S):
    Sb = band.to_band(S, p)
    assert band.to_band(S, p, False) == approx(band.to_upper(Sb))
    Sb = band.to_band(S, p, False)
    assert band.to_band(S, p) == approx(band.to_lower(Sb))


def test_cholesky(S):
    Sb = band.to_band(S, p, True)
    assert band.cholesky(Sb, True) == approx(linalg.cholesky_banded(Sb, lower=True))
    Sb = band.to_band(S, p, False)
    assert band.to_upper(band.cholesky(Sb, False)) == approx(
        linalg.cholesky_banded(Sb, lower=False)
    )


def test_solve_cholesky(S):
    Cb = band.cholesky(band.to_band(S, p, True), True)
    rhs = sps.rand(1, n, density=1, random_state=42).toarray().flatten()
    assert band.backward_sub(Cb, band.forward_sub(Cb, rhs, True), True) == approx(
        linalg.cho_solve_banded((Cb, True), rhs), rel=1e-4
    )

    Cb = band.to_upper(Cb)
    rhs = sps.rand(1, n, density=1, random_state=42).toarray().flatten()
    assert band.backward_sub(Cb, band.forward_sub(Cb, rhs, False), False) == approx(
        linalg.cho_solve_banded((Cb, False), rhs), rel=1e-4
    )


def test_solve_cholesky_permuted(S):
    rhs = sps.rand(1, n, density=1, random_state=42).toarray().flatten()
    Sp, P = band.permute(S)
    Sb = band.to_band(Sp, int(band.bandwidth(Sp)), True)
    assert Sb.shape[0] <= p + 1
    Cb = band.cholesky(Sb)
    x = P.T @ band.backward_sub(Cb, band.forward_sub(Cb, P @ rhs, True), True)
    assert x == approx(linalg.solve(S, rhs, assume_a="pos"), rel=1e-4)
    assert x == approx(
        linalg.solveh_banded(band.to_band(S, p), rhs, lower=True), rel=1e-4
    )
