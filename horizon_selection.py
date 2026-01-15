# -*- coding: utf-8 -*-
"""Horizon selection primitives.

This module bundles:
  - Information-form propagator horizon sweep (our method)
  - One-pass horizon selection helpers (baseline2)

Robustness updates (v3)
-----------------------
The one-pass baseline repeatedly solves systems with the regularized control
Hessian Q_uu. In earlier versions, a Cholesky failure fell back to SVD
(`np.linalg.lstsq`), which can crash with
"SVD did not converge" when matrices are ill-conditioned or contain NaNs/Infs.

This version:
- increases regularization (LM) locally until Q_uu is PD, and
- raises a controlled exception if it still cannot proceed.

The solver/runner catch these exceptions and mark the trial as a numerical
failure instead of terminating the whole benchmark.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from utils import _sym, as_terminal_weight, chol_inv, chol_solve, wrap_error


# =============================================================================
# Our method: information-form propagator sweep
# =============================================================================

def propagator_all_Jt_aug(
    A_aug: List[np.ndarray],
    B_aug: List[np.ndarray],
    Q_aug: List[np.ndarray],
    R_list: List[np.ndarray],
    z0: np.ndarray,
    QT_aug_list: List[np.ndarray],
    T_use: Optional[int] = None,
    R_inv_cached: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute J(T) for all T using an information-form propagator."""
    N = len(A_aug) if T_use is None else int(T_use)
    if N <= 0:
        return np.zeros(0, dtype=float)

    Rinvs = (
        [chol_inv(R_list[k]) for k in range(N)]
        if R_inv_cached is None
        else [R_inv_cached for _ in range(N)]
    )

    E_list, F_list, G_list = [], [], []
    for k in range(N):
        Ek = chol_inv(Q_aug[k])
        Fk = Ek @ A_aug[k].T
        Gk = A_aug[k] @ Ek @ A_aug[k].T + B_aug[k] @ Rinvs[k] @ B_aug[k].T
        E_list.append(Ek)
        F_list.append(Fk)
        G_list.append(_sym(Gk))

    # forward combine
    Ebar = [E_list[0].copy()]
    Fbar = [F_list[0].copy()]
    Gbar = [G_list[0].copy()]
    for k in range(1, N):
        Ek, Fk, Gk = E_list[k], F_list[k], G_list[k]
        W = chol_inv(Ek + Gbar[-1])
        Ebar.append(_sym(Ebar[-1] - Fbar[-1] @ W @ Fbar[-1].T))
        Fbar.append(Fbar[-1] @ W @ Fk)
        Gbar.append(_sym(Gk - Fk.T @ W @ Fk))

    # terminal sweeps
    J = np.zeros(N, dtype=float)
    QT_inv = [chol_inv(QT_aug_list[t - 1]) for t in range(1, N + 1)]
    for t in range(1, N + 1):
        Xt = QT_inv[t - 1]
        Wt = chol_inv(Xt + Gbar[t - 1])
        X0 = _sym(Ebar[t - 1] - Fbar[t - 1] @ Wt @ Fbar[t - 1].T)
        P0 = chol_inv(X0)
        J[t - 1] = 0.5 * float((z0.T @ P0 @ z0).item())
    return J


# =============================================================================
# Baseline2: single backward sweep value expansions
# =============================================================================

def _finite(x) -> bool:
    return bool(np.all(np.isfinite(x)))


def value_expansions_and_gains_prefix(
    A_list: List[np.ndarray],
    B_list: List[np.ndarray],
    X: np.ndarray,
    U: np.ndarray,
    xg: np.ndarray,
    u_ref: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    alpha,
    T_bar: int,
    S_right: int,
    *,
    lm_lambda: float = 1e-6,
    w_stage: float = 0.0,
    wrap_idx: Optional[List[int]] = None,
    extra_stage_cost=None,
    reg_max_tries: int = 12,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[np.ndarray], List[np.ndarray]]:
    """One backward sweep for t in [-S_right .. T_bar].

    Returns arrays with index shift +S_right:
      index i corresponds to time t = i - S_right.

    Robustness:
    - If any required quantity becomes non-finite, raise FloatingPointError.
    - If Q_uu is not PD, increase regularization locally until PD.
    """
    n, m = X.shape[1], U.shape[1]
    Qf = as_terminal_weight(alpha, n)

    L = int(T_bar) + int(S_right)
    Vxx = [np.zeros((n, n)) for _ in range(L + 1)]
    Vx = [np.zeros(n) for _ in range(L + 1)]
    V0 = [0.0 for _ in range(L + 1)]
    K = [None] * L
    k = [None] * L

    # terminal at t=T_bar => iT=T_bar+S_right
    iT = int(T_bar) + int(S_right)
    eT = np.atleast_1d(wrap_error(X[iT] - xg, wrap_idx)).reshape(-1)
    if not _finite(eT):
        raise FloatingPointError("Non-finite terminal error eT")

    Vxx[iT] = _sym(Qf)
    Vx[iT] = Qf @ eT
    V0[iT] = 0.5 * float(eT @ (Qf @ eT))

    # backward: t = T_bar-1 .. -S_right
    for t in reversed(range(-int(S_right), int(T_bar))):
        i = t + int(S_right)

        e = np.atleast_1d(wrap_error(X[i] - xg, wrap_idx)).reshape(-1)
        du = np.atleast_1d(U[i] - u_ref).reshape(-1)
        if not (_finite(e) and _finite(du)):
            raise FloatingPointError(f"Non-finite e/du at t={t} (i={i})")

        lx = Q @ e
        lu = R @ du
        l0 = 0.5 * float(e @ (Q @ e)) + 0.5 * float(du @ (R @ du)) + float(w_stage)
        Qstage = Q

        if extra_stage_cost is not None:
            c_extra, cx_extra, cxx_extra = extra_stage_cost(X[i], U[i])
            l0 += float(c_extra)
            lx = lx + np.asarray(cx_extra, dtype=float).reshape(-1)
            Qstage = _sym(Qstage + np.asarray(cxx_extra, dtype=float))

        A = np.asarray(A_list[i], dtype=float)
        B = np.asarray(B_list[i], dtype=float)
        if not (_finite(A) and _finite(B) and _finite(Vx[i + 1]) and _finite(Vxx[i + 1])):
            raise FloatingPointError(f"Non-finite A/B/V at t={t} (i={i})")

        Qx = lx + A.T @ Vx[i + 1]
        Qu = lu + B.T @ Vx[i + 1]
        Qxx = Qstage + A.T @ Vxx[i + 1] @ A
        Quu = R + B.T @ Vxx[i + 1] @ B
        Qux = B.T @ Vxx[i + 1] @ A

        if not (_finite(Qx) and _finite(Qu) and _finite(Qxx) and _finite(Quu) and _finite(Qux)):
            raise FloatingPointError(f"Non-finite Q terms at t={t} (i={i})")

        # Robust regularization loop (local LM)
        lam = float(max(lm_lambda, 1e-12))
        solved = False
        last_err: Optional[Exception] = None
        for _try in range(int(reg_max_tries)):
            Quu_reg = _sym(Quu) + lam * np.eye(m)
            if not _finite(Quu_reg):
                raise FloatingPointError(f"Non-finite Quu_reg at t={t} (i={i}), lam={lam:g}")
            try:
                invQuuQu = chol_solve(Quu_reg, Qu)
                invQuuQux = chol_solve(Quu_reg, Qux)
                solved = True
                break
            except Exception as e:
                last_err = e
                lam *= 10.0

        if not solved:
            raise np.linalg.LinAlgError(
                f"Failed to solve Quu system at t={t} (i={i}). "
                f"lm started={lm_lambda:g}, tried up to lam={lam:g}. Last err: {last_err}"
            )

        k[i] = -invQuuQu
        K[i] = -invQuuQux

        Vxx[i] = _sym(Qxx - Qux.T @ invQuuQux)
        Vx[i] = Qx - Qux.T @ invQuuQu
        V0[i] = l0 + V0[i + 1] - 0.5 * float(Qu.T @ invQuuQu)

        if not (_finite(Vxx[i]) and _finite(Vx[i]) and np.isfinite(V0[i])):
            raise FloatingPointError(f"Non-finite V at t={t} (i={i})")

    return Vxx, Vx, V0, K, k


def onepass_pick_T_singlepass(
    Vxx: List[np.ndarray],
    Vx: List[np.ndarray],
    V0: List[float],
    X_ext: np.ndarray,
    x0: np.ndarray,
    T_bar: int,
    T_min: int,
    T_max: int,
    S_left: int,
    S_right: int,
    wrap_idx: Optional[List[int]],
    *,
    locality_mult: float = 5.0,
) -> Tuple[int, np.ndarray]:
    """Pick T* in a window around T_bar using the quadratic value approximation.

    Candidate order is center-out (bidirectional): T̄, T̄±1, T̄±2, ...

    We also apply a mild locality gate to reduce gross mis-selections when the
    local quadratic approximation is invalid far from the nominal.
    """
    T_bar = int(T_bar)
    T_min = int(T_min)
    T_max = int(T_max)
    S_left = int(S_left)
    S_right = int(S_right)

    Jw = np.full(T_max, np.nan, dtype=float)

    L = max(T_min, T_bar - S_left)
    R = min(T_max, T_bar + S_right)
    if L > R:
        return int(np.clip(T_bar, T_min, T_max)), Jw

    cand = []
    norms = []
    for T in range(L, R + 1):
        t0 = T_bar - T
        i = t0 + S_right
        if i < 0 or i >= len(X_ext):
            continue
        dx0 = wrap_error((x0 - X_ext[i]).reshape(-1), wrap_idx)
        dn = float(np.linalg.norm(dx0))
        cand.append((T, i, dx0, dn))
        if np.isfinite(dn) and dn > 1e-12:
            norms.append(dn)

    if len(norms) > 0:
        ref = float(np.median(norms))
        dx_max = float(locality_mult) * ref
    else:
        dx_max = np.inf

    cand.sort(key=lambda it: (abs(it[0] - T_bar), it[0]))

    bestJ, bestT = np.inf, int(np.clip(T_bar, L, R))
    for T, i, dx0, dn in cand:
        if dn > dx_max:
            continue
        JT = 0.5 * float(dx0 @ (Vxx[i] @ dx0)) + float(Vx[i] @ dx0) + float(V0[i])
        Jw[T - 1] = JT
        if JT < bestJ:
            bestJ, bestT = JT, int(T)

    if not np.isfinite(bestJ):
        bestT = int(np.clip(T_bar, L, R))
    return bestT, Jw
