# -*- coding: utf-8 -*-
"""Utilities (linear algebra + angle wrapping).

This module is intentionally small and dependency-free.

Numerical robustness note
------------------------
Older versions used `np.linalg.lstsq` (SVD) as a fallback when Cholesky failed
(e.g., for Q_uu in the backward pass). When matrices become ill-conditioned or
contain NaNs/Infs, that SVD fallback can crash with:

  LinAlgError: SVD did not converge in Linear Least Squares

To avoid hard crashes, `chol_solve` here:
- tries Cholesky with increasing diagonal jitter, and
- if it still fails, raises `LinAlgError` so callers can increase
  regularization / reject / fall back cleanly.

This also eliminates LAPACK warnings such as
"On entry to DLASCL parameter number 4 had an illegal value" that often appear
when SVD is fed invalid values.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


# =============================================================================
# Small helpers
# =============================================================================

def _sym(A: np.ndarray) -> np.ndarray:
    """Symmetrize a matrix."""
    return 0.5 * (A + A.T)


def _assert_finite(name: str, X: np.ndarray):
    if not np.all(np.isfinite(X)):
        raise FloatingPointError(f"Non-finite values in {name}")


# =============================================================================
# Terminal weight helper
# =============================================================================

def as_terminal_weight(alpha, n: int) -> np.ndarray:
    """Convert scalar/diag/vector/matrix terminal weight into an (n,n) matrix."""
    A = np.asarray(alpha, dtype=float)
    if A.ndim == 0:
        return float(A) * np.eye(n)
    if A.ndim == 1:
        if A.shape[0] != n:
            raise ValueError(f"terminal weight vector has shape {A.shape}, expected ({n},)")
        return np.diag(A)
    if A.ndim == 2:
        if A.shape != (n, n):
            raise ValueError(f"terminal weight matrix has shape {A.shape}, expected ({n},{n})")
        return _sym(A)
    raise ValueError(f"unsupported terminal weight ndim={A.ndim}")


# =============================================================================
# Cholesky-based linear algebra
# =============================================================================

def chol_inv(A: np.ndarray, jitter: float = 1e-9, max_tries: int = 8) -> np.ndarray:
    """Inverse of a symmetric matrix using Cholesky + jitter.

    If Cholesky fails for all jitters, we fall back to an LU solve on (A+eps I).
    """
    A = _sym(np.asarray(A, dtype=float))
    _assert_finite("chol_inv(A)", A)

    n = A.shape[0]
    I = np.eye(n)
    eps = float(jitter)

    for _ in range(int(max_tries)):
        try:
            L = np.linalg.cholesky(A + eps * I)
            Y = np.linalg.solve(L, I)
            return np.linalg.solve(L.T, Y)
        except np.linalg.LinAlgError:
            eps *= 10.0

    # Deterministic fallback (avoids SVD)
    try:
        return np.linalg.solve(A + eps * I, I)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(f"chol_inv failed even with jitter={eps:g}: {e}")


def chol_solve(A: np.ndarray, B: np.ndarray, jitter: float = 1e-9, max_tries: int = 8) -> np.ndarray:
    """Solve A X = B for symmetric A using Cholesky + jitter.

    IMPORTANT: No SVD fallback. If Cholesky keeps failing, this raises.
    """
    A = _sym(np.asarray(A, dtype=float))
    B = np.asarray(B, dtype=float)
    _assert_finite("chol_solve(A)", A)
    _assert_finite("chol_solve(B)", B)

    n = A.shape[0]
    I = np.eye(n)
    eps = float(jitter)

    for _ in range(int(max_tries)):
        try:
            L = np.linalg.cholesky(A + eps * I)
            Y = np.linalg.solve(L, B)
            X = np.linalg.solve(L.T, Y)
            _assert_finite("chol_solve(X)", X)
            return X
        except (np.linalg.LinAlgError, FloatingPointError):
            eps *= 10.0

    raise np.linalg.LinAlgError(f"chol_solve failed: matrix not PD after jitter up to {eps:g}")


# =============================================================================
# Angle wrapping
# =============================================================================

def angle_normalize(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def wrap_error(e: np.ndarray, wrap_idx: Optional[List[int]] = None) -> np.ndarray:
    if not wrap_idx:
        return e
    e = np.asarray(e, dtype=float).copy()
    for i in wrap_idx:
        e[i] = angle_normalize(float(e[i]))
    return e
