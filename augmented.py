# -*- coding: utf-8 -*-
"""Augmented blocks used by the propagator horizon selection."""

import numpy as np
from typing import List, Optional

from utils import _sym, chol_inv, wrap_error, as_terminal_weight
from linearization import compute_affine_residuals

def build_augmented_sequence_QR(
    F, A_list, B_list, X, U, xg, u_ref, Q, R, w,
    wrap_idx: Optional[List[int]] = None,
    q_reg: float = 1e-9,
    rho_reg: float = 1e-12,
    extra_stage_cost=None,
):
    """Construct the augmented (A,B,Q,R) blocks for the propagator method.

    `extra_stage_cost` optionally adds a second-order Taylor approximation of an
    additional running state-cost (e.g., obstacle penalties).
    """
    N = len(A_list); n = X.shape[1]; m = U.shape[1]
    R = _sym(R); R_inv = chol_inv(R)
    a_list = compute_affine_residuals(F, X, U)
    A_aug, B_aug, Q_aug, R_list = [], [], [], []
    I_n = np.eye(n)
    for k in range(N):
        e  = wrap_error((X[k] - xg), wrap_idx).reshape(-1, 1)
        du = np.atleast_1d(U[k] - u_ref).reshape(-1, 1)

        Qk = np.zeros((n + 1, n + 1), dtype=float)
        Qk[:n, :n] = _sym(Q) + q_reg * I_n

        # quadratic Taylor expansion of 0.5*(e+dx)^T Q (e+dx) + w  around dx=0
        Qk[:n,  n] = (Q @ e).ravel()
        Qk[ n, :n] = (Q @ e).ravel()
        Qk[ n,  n] = float(e.T @ Q @ e) + 2.0 * float(w) + rho_reg

        if extra_stage_cost is not None:
            c_extra, cx_extra, cxx_extra = extra_stage_cost(X[k], U[k])
            cx_extra = np.asarray(cx_extra, dtype=float).reshape(-1)
            cxx_extra = _sym(np.asarray(cxx_extra, dtype=float))
            Qk[:n, :n] += cxx_extra
            Qk[:n,  n] += cx_extra
            Qk[ n, :n] += cx_extra
            Qk[ n,  n] += 2.0 * float(c_extra)

        Qk = _sym(Qk)

        atil = a_list[k] - B_list[k] @ du
        Ak_aug = np.zeros((n + 1, n + 1), dtype=float)
        Ak_aug[:n, :n] = A_list[k]
        Ak_aug[:n,  n] = atil.ravel()
        Ak_aug[ n,  n] = 1.0
        Bk_aug = np.zeros((n + 1, m), dtype=float)
        Bk_aug[:n, :] = B_list[k]
        A_aug.append(Ak_aug); B_aug.append(Bk_aug); Q_aug.append(Qk); R_list.append(R)

    z0 = np.zeros(n + 1); z0[-1] = 1.0
    return A_aug, B_aug, Q_aug, R_list, z0, R_inv


def build_terminal_aug_list(
    X, xg, alpha, wrap_idx: Optional[List[int]] = None, rho_reg: float = 1e-12
):
    """Terminal cost blocks for each candidate horizon.

    The propagator needs a terminal quadratic form at every possible timestep.
    `alpha` can be a scalar / diag vector / matrix terminal weight.
    """
    n = X.shape[1]
    Qf = as_terminal_weight(alpha, n)

    N = X.shape[0] - 1
    QT = []
    P = _sym(Qf)
    for t in range(1, N + 1):
        e  = wrap_error((X[t] - xg), wrap_idx).reshape(-1, 1)
        px = P @ e
        p0 = 0.5 * float((e.T @ (P @ e)).item())
        Qt = np.zeros((n + 1, n + 1), dtype=float)
        Qt[:-1, :-1] = P
        Qt[:-1, -1]  = px.ravel()
        Qt[-1,  :-1] = px.ravel()
        Qt[-1,  -1]  = 2.0 * p0 + rho_reg
        QT.append(_sym(Qt))
    return QT
