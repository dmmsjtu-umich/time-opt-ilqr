# -*- coding: utf-8 -*-
"""Time-optimal iLQR with horizon selection.

We support three solver variants:

- method="propagator"  (ourmethod)
    Uses an information-form propagator to compute J(T) for all candidate horizons.

- method="bruteforce"  (baseline1)
    Computes the exact quadratic-model J(T) curve via backward expansion (slow).

- method="onepass"     (baseline2)
    Paper-style one-pass horizon selection using a single backward sweep around T̄.

The core iLQR update (backward pass + forward line-search) is shared.
"""

from __future__ import annotations

import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from utils import _sym, chol_solve, wrap_error, as_terminal_weight
from linearization import (
    extend_nominal_backward,
    linearize_central_diff_traj,
    linearize_forward_diff_traj,
)
from augmented import build_augmented_sequence_QR, build_terminal_aug_list
from horizon_selection import (
    propagator_all_Jt_aug,
    value_expansions_and_gains_prefix,
    onepass_pick_T_singlepass,
)


# =============================================================================
# Rollout & objective
# =============================================================================

def rollout(F, x0: np.ndarray, U: np.ndarray, *, max_state_norm: float = 1e6) -> np.ndarray:
    """Roll forward dynamics with simple divergence checks."""
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U.reshape(-1, 1)

    N = U.shape[0]
    n = x0.size
    X = np.zeros((N + 1, n), dtype=float)
    X[0] = x0

    for k in range(N):
        xn = F(X[k], U[k])
        xn = np.asarray(xn, dtype=float).reshape(-1)
        if xn.size != n or (not np.all(np.isfinite(xn))) or (float(np.linalg.norm(xn)) > max_state_norm):
            X[k + 1 :] = np.nan
            break
        X[k + 1] = xn

    return X


def cost_timeopt_true(
    X: np.ndarray,
    U: np.ndarray,
    xg: np.ndarray,
    u_ref: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    alpha,
    w: float,
    T_star: int,
    wrap_idx: Optional[List[int]] = None,
    extra_stage_cost=None,
) -> float:
    """True objective: running cost up to T_star + terminal cost at T_star."""
    T_star = int(T_star)
    if T_star <= 0:
        return float("inf")

    if (not np.all(np.isfinite(X[: T_star + 1]))) or (not np.all(np.isfinite(U[:T_star]))):
        return float("inf")

    n = X.shape[1]
    Qf = as_terminal_weight(alpha, n)

    c = 0.0
    for k in range(T_star):
        e = np.atleast_1d(wrap_error(X[k] - xg, wrap_idx)).reshape(-1)
        du = np.atleast_1d(U[k] - u_ref).reshape(-1)
        # guard against numerical blow-up (matmul overflow)
        if (not np.all(np.isfinite(e))) or (not np.all(np.isfinite(du))):
            return float("inf")
        c += 0.5 * float(e @ (Q @ e)) + 0.5 * float(du @ (R @ du)) + float(w)
        if extra_stage_cost is not None:
            c_extra, _, _ = extra_stage_cost(X[k], U[k])
            c += float(c_extra)

    eT = np.atleast_1d(wrap_error(X[T_star] - xg, wrap_idx)).reshape(-1)
    if not np.all(np.isfinite(eT)):
        return float("inf")
    c += 0.5 * float(eT @ (Qf @ eT))
    return float(c)


def nominal_cost_curve(
    X: np.ndarray,
    U: np.ndarray,
    xg: np.ndarray,
    u_ref: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    alpha,
    w: float,
    T_min: int,
    T_max: int,
    wrap_idx: Optional[List[int]] = None,
    extra_stage_cost=None,
) -> np.ndarray:
    """Cost curve J_nom(T) for the *current nominal* trajectory (X,U).

    This is only used as a cheap heuristic to pick an initial T̄ for baseline2.
    """
    T_min = int(T_min)
    T_max = int(T_max)
    n = X.shape[1]
    Qf = as_terminal_weight(alpha, n)

    J = np.full(T_max, np.inf, dtype=float)
    if not np.all(np.isfinite(X[: T_max + 1])) or not np.all(np.isfinite(U[:T_max])):
        return J

    running = 0.0
    for k in range(1, T_max + 1):
        # add stage cost at k-1
        e = np.atleast_1d(wrap_error(X[k - 1] - xg, wrap_idx)).reshape(-1)
        du = np.atleast_1d(U[k - 1] - u_ref).reshape(-1)
        running += 0.5 * float(e @ (Q @ e)) + 0.5 * float(du @ (R @ du)) + float(w)
        if extra_stage_cost is not None:
            c_extra, _, _ = extra_stage_cost(X[k - 1], U[k - 1])
            running += float(c_extra)

        if k >= T_min:
            eT = np.atleast_1d(wrap_error(X[k] - xg, wrap_idx)).reshape(-1)
            J[k - 1] = float(running + 0.5 * float(eT @ (Qf @ eT)))

    return J


# =============================================================================
# Fixed-horizon iLQR primitives (backward + forward)
# =============================================================================

def backward_pass_truncated(
    A_list: List[np.ndarray],
    B_list: List[np.ndarray],
    X: np.ndarray,
    U: np.ndarray,
    xg: np.ndarray,
    u_ref: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    alpha,
    T_star: int,
    *,
    lm_lambda: float = 1e-3,
    wrap_idx: Optional[List[int]] = None,
    extra_stage_cost=None,
) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]], bool]:
    """Standard iLQR backward pass on a fixed horizon [0..T_star]."""
    T_star = int(T_star)
    if T_star <= 0:
        return None, None, False

    n, m = X.shape[1], U.shape[1]
    Qf = as_terminal_weight(alpha, n)

    k_list = [None] * T_star
    K_list = [None] * T_star

    eT = np.atleast_1d(wrap_error(X[T_star] - xg, wrap_idx)).reshape(-1)
    if not np.all(np.isfinite(eT)):
        return None, None, False
    Vx = Qf @ eT
    Vxx = _sym(Qf)

    for k in reversed(range(T_star)):
        e = np.atleast_1d(wrap_error(X[k] - xg, wrap_idx)).reshape(-1)
        du = np.atleast_1d(U[k] - u_ref).reshape(-1)
        if not np.all(np.isfinite(e)) or not np.all(np.isfinite(du)):
            return None, None, False

        lx = Q @ e
        lu = R @ du
        Qstage = Q

        if extra_stage_cost is not None:
            _, cx_extra, cxx_extra = extra_stage_cost(X[k], U[k])
            lx = lx + np.asarray(cx_extra, dtype=float).reshape(-1)
            Qstage = _sym(Qstage + np.asarray(cxx_extra, dtype=float))

        A, B = A_list[k], B_list[k]
        Qx = lx + A.T @ Vx
        Qu = lu + B.T @ Vx
        Qxx = Qstage + A.T @ Vxx @ A
        Quu = R + B.T @ Vxx @ B
        Qux = B.T @ Vxx @ A

        Quu_reg = _sym(Quu) + float(lm_lambda) * np.eye(m)
        # SPD check
        try:
            np.linalg.cholesky(Quu_reg)
        except np.linalg.LinAlgError:
            return None, None, False

        kappa = -chol_solve(Quu_reg, Qu)
        Kk = -chol_solve(Quu_reg, Qux)

        k_list[k] = kappa
        K_list[k] = Kk

        Vx = Qx + Kk.T @ Qu + Qux.T @ kappa + Kk.T @ Quu @ kappa
        Vxx = _sym(Qxx + Kk.T @ Qux + Qux.T @ Kk + Kk.T @ Quu @ Kk)

        if not np.all(np.isfinite(Vx)) or not np.all(np.isfinite(Vxx)):
            return None, None, False

    return k_list, K_list, True


def forward_linesearch_fixedT(
    F,
    X: np.ndarray,
    U: np.ndarray,
    xg: np.ndarray,
    u_ref: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    alpha,
    w: float,
    T_star: int,
    k_list: List[np.ndarray],
    K_list: List[np.ndarray],
    *,
    alphas: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05),
    wrap_idx: Optional[List[int]] = None,
    extra_stage_cost=None,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """Forward pass with line-search (fixed horizon T_star)."""
    T_star = int(T_star)
    J_old = cost_timeopt_true(X, U, xg, u_ref, Q, R, alpha, w, T_star, wrap_idx, extra_stage_cost)

    N = len(U)
    for a in alphas:
        U_new = U.copy()
        X_new = np.zeros_like(X)
        X_new[0] = X[0]

        ok = True
        for k in range(T_star):
            dx = wrap_error((X_new[k] - X[k]).reshape(-1), wrap_idx)
            du = (K_list[k] @ dx + float(a) * k_list[k]).reshape(-1)
            U_new[k] = (U[k] + du).reshape(-1)
            X_new[k + 1] = F(X_new[k], U_new[k])
            if not np.all(np.isfinite(X_new[k + 1])):
                ok = False
                break

        if not ok:
            continue

        for k in range(T_star, N):
            X_new[k + 1] = F(X_new[k], U_new[k])
            if not np.all(np.isfinite(X_new[k + 1])):
                ok = False
                break
        if not ok:
            continue

        J_new = cost_timeopt_true(X_new, U_new, xg, u_ref, Q, R, alpha, w, T_star, wrap_idx, extra_stage_cost)
        if J_new < J_old:
            return X_new, U_new, J_new, True

    return X, U, J_old, False


# =============================================================================
# Baseline1: brute-force quadratic-model J(T) curve
# =============================================================================

def bruteforce_all_Jt_backward_expansion(
    A_list: List[np.ndarray],
    B_list: List[np.ndarray],
    X: np.ndarray,
    U: np.ndarray,
    xg: np.ndarray,
    u_ref: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    alpha,
    w: float,
    T_max: int,
    *,
    lm_lambda: float = 1e-6,
    wrap_idx: Optional[List[int]] = None,
    extra_stage_cost=None,
) -> np.ndarray:
    """Exact J(T) curve under the iLQR quadratic model (slow)."""
    T_max = int(T_max)
    n, m = X.shape[1], U.shape[1]
    Qf = as_terminal_weight(alpha, n)

    J = np.zeros(T_max, dtype=float)
    for T in range(1, T_max + 1):
        Vxx = [np.zeros((n, n)) for _ in range(T + 1)]
        Vx  = [np.zeros(n) for _ in range(T + 1)]
        V0  = [0.0 for _ in range(T + 1)]

        eT = np.atleast_1d(wrap_error(X[T] - xg, wrap_idx)).reshape(-1)
        Vxx[T] = _sym(Qf)
        Vx[T] = Qf @ eT
        V0[T] = 0.5 * float(eT @ (Qf @ eT))

        for t in reversed(range(T)):
            e = np.atleast_1d(wrap_error(X[t] - xg, wrap_idx)).reshape(-1)
            du = np.atleast_1d(U[t] - u_ref).reshape(-1)

            lx = Q @ e
            lu = R @ du
            l0 = 0.5 * float(e @ (Q @ e)) + 0.5 * float(du @ (R @ du)) + float(w)
            Qstage = Q

            if extra_stage_cost is not None:
                c_extra, cx_extra, cxx_extra = extra_stage_cost(X[t], U[t])
                l0 += float(c_extra)
                lx = lx + np.asarray(cx_extra, dtype=float).reshape(-1)
                Qstage = _sym(Qstage + np.asarray(cxx_extra, dtype=float))

            A, B = A_list[t], B_list[t]
            Qx = lx + A.T @ Vx[t + 1]
            Qu = lu + B.T @ Vx[t + 1]
            Qxx = Qstage + A.T @ Vxx[t + 1] @ A
            Quu = R + B.T @ Vxx[t + 1] @ B
            Qux = B.T @ Vxx[t + 1] @ A

            Quu_reg = _sym(Quu) + float(lm_lambda) * np.eye(m)
            invQuuQu  = chol_solve(Quu_reg, Qu)
            invQuuQux = chol_solve(Quu_reg, Qux)

            Vxx[t] = _sym(Qxx - Qux.T @ invQuuQux)
            Vx[t]  = Qx  - Qux.T @ invQuuQu
            V0[t]  = l0  + V0[t + 1] - 0.5 * float(Qu.T @ invQuuQu)

        J[T - 1] = float(V0[0])

    return J


# =============================================================================
# Baseline2: one-pass rollout
# =============================================================================

def onepass_rollout(
    F,
    X_ext: np.ndarray,
    U_ext: np.ndarray,
    xg: np.ndarray,
    u_ref: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    alpha,
    w: float,
    K: List[np.ndarray],
    k: List[np.ndarray],
    *,
    T_bar: int,
    T_star: int,
    S_right: int,
    wrap_idx: Optional[List[int]] = None,
    extra_stage_cost=None,
    alphas: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.1),
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """Rollout the one-pass policy for a candidate (T_star) with line-search.

    The control law uses the gains from the single backward sweep around T̄:
      u_t = ū_{t0+t} + K_{t0+t} (x_t - x̄_{t0+t}) + α k_{t0+t}
      where t0 = T̄ - T*.

    Returns (X_new, U_new, J_new, ok_improved) where ok_improved only indicates
    finite rollout; the caller compares J_new with the previous cost.
    """
    T_bar = int(T_bar)
    T_star = int(T_star)
    S_right = int(S_right)

    t0 = T_bar - T_star
    off = S_right

    # original horizon length
    N = U_ext.shape[0] - off
    n = X_ext.shape[1]

    J_best = float("inf")
    X_best = None
    U_best = None

    for a in alphas:
        Xn = np.zeros((N + 1, n), dtype=float)
        Un = U_ext[off:].copy()
        Xn[0] = X_ext[off]

        ok = True
        for t in range(T_star):
            idx = t0 + t + off
            dx = wrap_error((Xn[t] - X_ext[idx]).reshape(-1), wrap_idx)
            du = (K[idx] @ dx + float(a) * k[idx]).reshape(-1)
            Un[t] = (U_ext[idx] + du).reshape(-1)
            Xn[t + 1] = F(Xn[t], Un[t])
            if not np.all(np.isfinite(Xn[t + 1])):
                ok = False
                break
        if not ok:
            continue

        for t in range(T_star, N):
            Xn[t + 1] = F(Xn[t], Un[t])
            if not np.all(np.isfinite(Xn[t + 1])):
                ok = False
                break
        if not ok:
            continue

        Jn = cost_timeopt_true(Xn, Un, xg, u_ref, Q, R, alpha, w, T_star, wrap_idx, extra_stage_cost)
        if Jn < J_best:
            J_best, X_best, U_best = Jn, Xn, Un

    if X_best is None:
        return X_ext[off:].copy(), U_ext[off:].copy(), float("inf"), False

    return X_best, U_best, float(J_best), True


# =============================================================================
# Main solver
# =============================================================================

def ilqr_timeopt(
    F,
    x0: np.ndarray,
    xg: np.ndarray,
    u_ref: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    alpha,
    w: float,
    N: int,
    T_min: int,
    T_max: int,
    *,
    U_init: Optional[np.ndarray] = None,
    method: str = "propagator",
    max_iter: int = 15,
    lm_init: float = 1e-3,
    S_window: int = 20,
    wrap_idx: Optional[List[int]] = None,
    use_central_diff: bool = True,
    extra_stage_cost=None,
    onepass_preimage: str = "fixedpoint",
) -> Dict[str, Any]:
    """Solve the time-penalized horizon-selection iLQR problem."""
    assert method in ("propagator", "bruteforce", "onepass")

    N = int(N)
    T_min = int(T_min)
    T_max = int(T_max)

    # initial nominal controls
    if U_init is None:
        U = np.tile(np.asarray(u_ref, dtype=float).reshape(1, -1), (N, 1))
    else:
        U = np.asarray(U_init, dtype=float)
        if U.ndim == 1:
            U = U.reshape(-1, 1)
        if U.shape[0] < N:
            pad = np.tile(U[-1:], (N - U.shape[0], 1))
            U = np.vstack([U, pad])
        elif U.shape[0] > N:
            U = U[:N]

    X = rollout(F, x0, U)

    J_hist: List[float] = []
    T_hist: List[int] = []

    timers = {"linearize": 0.0, "select": 0.0, "backward": 0.0, "forward": 0.0}
    lm = float(lm_init)
    onepass_last_error = None  # filled only for method="onepass"

    # -----------------------------
    # initial linearization (original trajectory)
    # -----------------------------
    t0 = time.perf_counter()
    if use_central_diff:
        A_list, B_list = linearize_central_diff_traj(F, X, U)
    else:
        A_list, B_list = linearize_forward_diff_traj(F, X, U)
    timers["linearize"] += time.perf_counter() - t0

    # -----------------------------
    # initial horizon guess T̄
    # -----------------------------
    if method == "propagator":
        t1 = time.perf_counter()
        A_aug, B_aug, Q_aug, R_list, z0, R_inv = build_augmented_sequence_QR(
            F, A_list, B_list, X, U, xg, u_ref, Q, R, w,
            wrap_idx=wrap_idx, extra_stage_cost=extra_stage_cost
        )
        QT_list = build_terminal_aug_list(X, xg, alpha, wrap_idx=wrap_idx)
        J_curve = propagator_all_Jt_aug(A_aug, B_aug, Q_aug, R_list, z0, QT_list, T_use=T_max, R_inv_cached=R_inv)
        T_bar = int(np.argmin(J_curve[T_min - 1 : T_max]) + T_min)
        timers["select"] += time.perf_counter() - t1

    elif method == "bruteforce":
        t1 = time.perf_counter()
        J_curve = bruteforce_all_Jt_backward_expansion(
            A_list, B_list, X, U, xg, u_ref, Q, R, alpha, w, T_max,
            wrap_idx=wrap_idx, extra_stage_cost=extra_stage_cost
        )
        T_bar = int(np.argmin(J_curve[T_min - 1 : T_max]) + T_min)
        timers["select"] += time.perf_counter() - t1

    else:  # onepass: pick a cheap initial guess from the nominal trajectory
        J_curve = None
        J_nom = nominal_cost_curve(X, U, xg, u_ref, Q, R, alpha, w, T_min, T_max, wrap_idx, extra_stage_cost)
        T_bar = int(np.argmin(J_nom[T_min - 1 : T_max]) + T_min)

    # warm-start update at T̄
    t2 = time.perf_counter()
    k_list, K_list, ok = backward_pass_truncated(
        A_list, B_list, X, U, xg, u_ref, Q, R, alpha, T_bar,
        lm_lambda=lm, wrap_idx=wrap_idx, extra_stage_cost=extra_stage_cost
    )
    timers["backward"] += time.perf_counter() - t2
    if ok:
        t3 = time.perf_counter()
        X, U, J0, _ = forward_linesearch_fixedT(
            F, X, U, xg, u_ref, Q, R, alpha, w, T_bar, k_list, K_list,
            wrap_idx=wrap_idx, extra_stage_cost=extra_stage_cost
        )
        timers["forward"] += time.perf_counter() - t3
        if np.isfinite(J0):
            J_hist.append(float(J0))
            T_hist.append(int(T_bar))

    S_left = int(S_window)
    S_right = int(S_window)
    last_window_curve = None

    # -----------------------------
    # outer loop
    # -----------------------------
    for _it in range(int(max_iter)):
        # 1) linearize
        t0 = time.perf_counter()
        if use_central_diff:
            A_list, B_list = linearize_central_diff_traj(F, X, U)
        else:
            A_list, B_list = linearize_forward_diff_traj(F, X, U)
        timers["linearize"] += time.perf_counter() - t0

        # 2) method-specific selection + forward update
        acc = False
        T_star = int(T_bar)
        Xn, Un, Jn = X, U, (J_hist[-1] if len(J_hist) else float("inf"))
        cur_J_prev = J_hist[-1] if len(J_hist) else float("inf")

        if method == "propagator":
            t1 = time.perf_counter()
            A_aug, B_aug, Q_aug, R_list, z0, R_inv = build_augmented_sequence_QR(
                F, A_list, B_list, X, U, xg, u_ref, Q, R, w,
                wrap_idx=wrap_idx, extra_stage_cost=extra_stage_cost
            )
            QT_list = build_terminal_aug_list(X, xg, alpha, wrap_idx=wrap_idx)
            J_curve = propagator_all_Jt_aug(
                A_aug, B_aug, Q_aug, R_list, z0, QT_list,
                T_use=T_max, R_inv_cached=R_inv
            )
            T_star = int(np.argmin(J_curve[T_min - 1 : T_max]) + T_min)
            timers["select"] += time.perf_counter() - t1

            t2 = time.perf_counter()
            k_list, K_list, ok = backward_pass_truncated(
                A_list, B_list, X, U, xg, u_ref, Q, R, alpha, T_star,
                lm_lambda=lm, wrap_idx=wrap_idx, extra_stage_cost=extra_stage_cost
            )
            timers["backward"] += time.perf_counter() - t2
            if ok:
                t3 = time.perf_counter()
                Xn, Un, Jn, acc = forward_linesearch_fixedT(
                    F, X, U, xg, u_ref, Q, R, alpha, w, T_star,
                    k_list, K_list, wrap_idx=wrap_idx, extra_stage_cost=extra_stage_cost
                )
                timers["forward"] += time.perf_counter() - t3

        elif method == "bruteforce":
            t1 = time.perf_counter()
            J_curve = bruteforce_all_Jt_backward_expansion(
                A_list, B_list, X, U, xg, u_ref, Q, R, alpha, w, T_max,
                wrap_idx=wrap_idx, extra_stage_cost=extra_stage_cost
            )
            T_star = int(np.argmin(J_curve[T_min - 1 : T_max]) + T_min)
            timers["select"] += time.perf_counter() - t1

            t2 = time.perf_counter()
            k_list, K_list, ok = backward_pass_truncated(
                A_list, B_list, X, U, xg, u_ref, Q, R, alpha, T_star,
                lm_lambda=lm, wrap_idx=wrap_idx, extra_stage_cost=extra_stage_cost
            )
            timers["backward"] += time.perf_counter() - t2
            if ok:
                t3 = time.perf_counter()
                Xn, Un, Jn, acc = forward_linesearch_fixedT(
                    F, X, U, xg, u_ref, Q, R, alpha, w, T_star,
                    k_list, K_list, wrap_idx=wrap_idx, extra_stage_cost=extra_stage_cost
                )
                timers["forward"] += time.perf_counter() - t3

        else:  # onepass
            # build negative-time prefix (cheap) ONLY for the window extension
            # Use U[0] as the fill control so the prefix is continuous with the nominal.
            u_fill = np.asarray(U[0], dtype=float).reshape(-1)

            t_lin = time.perf_counter()
            X_ext, U_ext = extend_nominal_backward(
                F, X, U, u_fill,
                S_back=S_right,
                preimage_method=onepass_preimage,
                max_iter=4,
                damping=0.5,
            )

            # Linearize prefix with *forward* differencing (fast) and reuse the
            # original (A_list,B_list) for the nonnegative segment.
            if S_right > 0:
                A_pre, B_pre = linearize_forward_diff_traj(
                    F,
                    X_ext[: S_right + 1],
                    U_ext[:S_right],
                )
                A_ext = A_pre + A_list
                B_ext = B_pre + B_list
            else:
                A_ext, B_ext = A_list, B_list
            timers["linearize"] += time.perf_counter() - t_lin

            t_sel = time.perf_counter()
            try:
                Vxx, Vx, V0, Kp, kp = value_expansions_and_gains_prefix(
                    A_ext, B_ext, X_ext, U_ext,
                    xg, u_ref, Q, R, alpha,
                    T_bar, S_right,
                    lm_lambda=lm,
                    w_stage=w,
                    wrap_idx=wrap_idx,
                    extra_stage_cost=extra_stage_cost,
                )
                T_star, Jw = onepass_pick_T_singlepass(
                    Vxx, Vx, V0, X_ext, X_ext[S_right], T_bar, T_min, T_max,
                    S_left, S_right, wrap_idx
                )
                last_window_curve = Jw
            except Exception as e:
                # Numerical failure in one-pass sweep: do NOT crash the whole run.
                # Fall back to a standard truncated iLQR update at the current T_bar.
                last_window_curve = None
                onepass_last_error = repr(e)
                T_star = int(T_bar)

                timers["select"] += time.perf_counter() - t_sel

                t2 = time.perf_counter()
                k_list, K_list, ok = backward_pass_truncated(
                    A_list, B_list, X, U, xg, u_ref, Q, R, alpha, T_star,
                    lm_lambda=lm,
                    wrap_idx=wrap_idx,
                    extra_stage_cost=extra_stage_cost,
                )
                timers["backward"] += time.perf_counter() - t2
                if ok:
                    t3 = time.perf_counter()
                    Xn, Un, Jn, acc = forward_linesearch_fixedT(
                        F, X, U, xg, u_ref, Q, R, alpha, w, T_star,
                        k_list, K_list,
                        wrap_idx=wrap_idx,
                        extra_stage_cost=extra_stage_cost,
                    )
                    timers["forward"] += time.perf_counter() - t3
                else:
                    acc = False

            else:
                timers["select"] += time.perf_counter() - t_sel

                t_fwd = time.perf_counter()
                # try a few window shrinks if rollout doesn't improve
                S_L_cur, S_R_cur = S_left, S_right
                for _shrink in range(3):
                    Xcand, Ucand, Jcand, ok_roll = onepass_rollout(
                        F, X_ext, U_ext, xg, u_ref, Q, R, alpha, w,
                        Kp, kp,
                        T_bar=T_bar,
                        T_star=T_star,
                        S_right=S_right,
                        wrap_idx=wrap_idx,
                        extra_stage_cost=extra_stage_cost,
                    )
                    if ok_roll and (Jcand < cur_J_prev):
                        Xn, Un, Jn, acc = Xcand, Ucand, Jcand, True
                        break

                    # shrink and re-pick
                    S_L_cur = max(1, S_L_cur // 2)
                    S_R_cur = max(1, S_R_cur // 2)
                    T_star, Jw = onepass_pick_T_singlepass(
                        Vxx, Vx, V0, X_ext, X_ext[S_right], T_bar, T_min, T_max,
                        S_L_cur, S_R_cur, wrap_idx
                    )
                    last_window_curve = Jw

                timers["forward"] += time.perf_counter() - t_fwd

        # 3) accept / reject
        if acc and np.isfinite(Jn):
            X, U = Xn, Un
            T_bar = int(T_star)
            J_hist.append(float(Jn))
            T_hist.append(int(T_star))
            lm = max(lm / 10.0, 1e-12)
        else:
            lm *= 10.0

        # 4) stop if stable
        if len(J_hist) >= 2:
            rel = abs(J_hist[-1] - J_hist[-2]) / (abs(J_hist[-2]) + 1e-12)
            if rel < 1e-4 and (len(T_hist) >= 3 and len(set(T_hist[-3:])) == 1):
                break

    # For plotting/debug: return the last computed selection curve
    if method in ("propagator", "bruteforce"):
        J_curve_out = J_curve
    else:
        J_curve_out = last_window_curve

    return {
        "X": X,
        "U": U,
        "J_hist": J_hist,
        "T_hist": T_hist,
        "timers": timers,
        "J_curve": J_curve_out,
        "T_star": int(T_hist[-1] if len(T_hist) else T_bar),
        "onepass_error": onepass_last_error,
    }


# =============================================================================
# Convenience wrappers (names used by run_suite / paper plots)
# =============================================================================

def ilqr_timeopt_ourmethod(*args, **kwargs):
    return ilqr_timeopt(*args, method="propagator", **kwargs)

def ilqr_timeopt_baseline1(*args, **kwargs):
    return ilqr_timeopt(*args, method="bruteforce", **kwargs)

def ilqr_timeopt_baseline2(*args, **kwargs):
    return ilqr_timeopt(*args, method="onepass", **kwargs)
