# -*- coding: utf-8 -*-
"""Linearization utilities + negative-time prefix helpers.

The "one-pass" baseline (paper-style horizon selection) may evaluate candidate
horizons T > T̄, which corresponds to negative start indices t0 = T̄ - T < 0.
The paper suggests constructing an arbitrary *dynamically feasible* negative-time
prefix (x_{-s}, u_{-s}) for s>0 (e.g., by integrating backwards).

A full Newton pre-image solve is often unnecessary and can dominate runtime.
We provide a much cheaper damped fixed-point preimage iteration, with Newton
kept as an optional debug/accuracy mode.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from utils import angle_normalize, wrap_error  # keep for compatibility


# =============================================================================
# Finite-diff Jacobian (w.r.t. x) – used only by Newton preimage
# =============================================================================

def jacobian_x_fd(F, x, u, eps: float = 1e-6) -> np.ndarray:
    n = x.size
    J = np.zeros((n, n))
    f0 = F(x, u)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        J[:, i] = (F(x + dx, u) - f0) / eps
    return J


# =============================================================================
# Negative-time prefix: preimage steps
# =============================================================================

def fixedpoint_preimage_step(
    F,
    x_next: np.ndarray,
    u_prev: np.ndarray,
    *,
    max_iter: int = 6,
    tol: float = 1e-9,
    damping: float = 0.5,
) -> np.ndarray:
    """Approximate solve of F(x_prev, u_prev) = x_next via damped fixed-point iteration.

    Iteration:
      x <- x - damping * (F(x,u) - x_next)

    This uses only 1 call to F per iteration (very fast), and is usually enough
    to construct a feasible-ish negative prefix for horizon selection.
    """
    x = np.asarray(x_next, dtype=float).copy()
    u_prev = np.asarray(u_prev, dtype=float).reshape(-1)

    for _ in range(int(max_iter)):
        fx = F(x, u_prev)
        if not np.all(np.isfinite(fx)):
            break
        r = fx - x_next
        nr = float(np.linalg.norm(r))
        if nr < tol:
            return x
        x = x - float(damping) * r

    return x


def newton_preimage_step(
    F,
    x_next: np.ndarray,
    u_prev: np.ndarray,
    *,
    max_iter: int = 10,
    tol: float = 1e-9,
) -> np.ndarray:
    """Solve F(x_prev, u_prev) = x_next via Newton (expensive)."""
    x = np.asarray(x_next, dtype=float).copy()
    u_prev = np.asarray(u_prev, dtype=float).reshape(-1)

    for _ in range(int(max_iter)):
        fx = F(x, u_prev)
        if not np.all(np.isfinite(fx)):
            break
        g = fx - x_next
        if float(np.linalg.norm(g)) < tol:
            return x
        J = jacobian_x_fd(F, x, u_prev)
        try:
            dx = np.linalg.solve(J, g)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(J, g, rcond=None)[0]
        # simple damping
        step = 1.0
        x_new = x - step * dx
        if not np.all(np.isfinite(x_new)):
            step = 0.5
            x_new = x - step * dx
        x = x_new

    return x


def extend_nominal_backward(
    F,
    X: np.ndarray,
    U: np.ndarray,
    u_fill: np.ndarray,
    *,
    S_back: int,
    preimage_method: str = "fixedpoint",
    max_iter: int = 6,
    tol: float = 1e-9,
    damping: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a negative-time prefix of length S_back.

    Returns:
      X_ext: shape (S_back + (N+1), n)
      U_ext: shape (S_back + N, m)

    The prefix uses constant control u_fill (shape (m,)) for all negative steps.

    preimage_method:
      - "fixedpoint": fast damped fixed-point (default)
      - "newton": Newton solve (slow; debug)
      - "copy": x_{-s} = x0 (only correct if (x0,u_fill) is a fixed point)
    """
    S_back = int(S_back)
    if S_back <= 0:
        return X.copy(), U.copy()

    preimage_method = str(preimage_method).lower().strip()
    u_fill = np.asarray(u_fill, dtype=float).reshape(-1)

    n = X.shape[1]
    m = U.shape[1]

    X_ext = np.zeros((X.shape[0] + S_back, n), dtype=float)
    U_ext = np.zeros((U.shape[0] + S_back, m), dtype=float)

    # original segment at the tail
    X_ext[S_back:] = X
    U_ext[S_back:] = U

    x_curr = np.asarray(X[0], dtype=float).copy()
    for s in range(1, S_back + 1):
        if preimage_method == "copy":
            x_prev = x_curr.copy()
        elif preimage_method == "newton":
            x_prev = newton_preimage_step(F, x_curr, u_fill, max_iter=max_iter, tol=tol)
        else:
            x_prev = fixedpoint_preimage_step(
                F, x_curr, u_fill, max_iter=max_iter, tol=tol, damping=damping
            )

        if not np.all(np.isfinite(x_prev)):
            # fall back: keep constant (still bounded; not necessarily feasible)
            x_prev = x_curr.copy()

        X_ext[S_back - s] = x_prev
        U_ext[S_back - s] = u_fill
        x_curr = x_prev

    return X_ext, U_ext


# =============================================================================
# Linearization along a nominal trajectory
# =============================================================================

def linearize_central_diff_traj(
    F,
    X: np.ndarray,
    U: np.ndarray,
    epsx: float = 1e-5,
    epsu: float = 1e-5,
    relx: float = 1e-6,
    relu: float = 1e-6,
):
    N = len(U)
    n = X.shape[1]
    m = U.shape[1]
    A_list, B_list = [], []
    for k in range(N):
        x = X[k].copy()
        u = U[k].copy()
        A = np.zeros((n, n))
        B = np.zeros((n, m))
        for i in range(n):
            hi = max(epsx, relx * max(1.0, abs(float(x[i]))))
            xp = x.copy()
            xm = x.copy()
            xp[i] += hi
            xm[i] -= hi
            A[:, i] = (F(xp, u) - F(xm, u)) / (2.0 * hi)
        for j in range(m):
            hj = max(epsu, relu * max(1.0, abs(float(u[j]))))
            up = u.copy()
            um = u.copy()
            up[j] += hj
            um[j] -= hj
            B[:, j] = (F(x, up) - F(x, um)) / (2.0 * hj)
        A_list.append(A)
        B_list.append(B)
    return A_list, B_list




def linearize_forward_diff_traj(
    F,
    X: np.ndarray,
    U: np.ndarray,
    epsx: float = 1e-5,
    epsu: float = 1e-5,
    relx: float = 1e-6,
    relu: float = 1e-6,
):
    """Forward-difference linearization with *relative* step sizes.

    The previous constant-step version (eps=1e-6) can be too small for strongly
    nonlinear systems (e.g., Euler-angle quadrotor), leading to noisy A,B and
    poor convergence. We instead use
      h_i = max(eps, rel * max(1, |x_i|))
    per dimension, which is much more stable.
    """
    N = len(U)
    n = X.shape[1]
    m = U.shape[1]
    A_list, B_list = [], []
    I_n, I_m = np.eye(n), np.eye(m)
    for k in range(N):
        x = np.asarray(X[k], dtype=float)
        u = np.asarray(U[k], dtype=float)
        f0 = F(x, u)
        A = np.zeros((n, n), dtype=float)
        B = np.zeros((n, m), dtype=float)

        if not np.all(np.isfinite(f0)):
            A[:] = np.nan
            B[:] = np.nan
            A_list.append(A)
            B_list.append(B)
            continue

        for i in range(n):
            hi = max(float(epsx), float(relx) * max(1.0, abs(float(x[i]))))
            A[:, i] = (F(x + hi * I_n[i], u) - f0) / hi

        for j in range(m):
            hj = max(float(epsu), float(relu) * max(1.0, abs(float(u[j]))))
            B[:, j] = (F(x, u + hj * I_m[j]) - f0) / hj

        A_list.append(A)
        B_list.append(B)
    return A_list, B_list


# =============================================================================
# Affine residuals for linearized dynamics
# =============================================================================

def compute_affine_residuals(F, X: np.ndarray, U: np.ndarray):
    return [(F(X[k], U[k]) - X[k + 1]).reshape(-1, 1) for k in range(len(U))]
