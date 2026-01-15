# -*- coding: utf-8 -*-
"""
Time-Optimal iLQR with three selection strategies:
  - 'propagator' : augmented information-form (fast, exact J_t under the quadratic model)
  - 'onepass'    : Algorithm 1 style: single backward at T̄, evaluate a single window [T̄-S_L, T̄+S_R]
                   using V_{t0:T̄} with a negative-time prefix built by backward feasibility (Newton).
  - 'bruteforce' : J_t via backward expansion ONLY (no warm-start sweep), then one fixed-T update
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Optional

# =============================================================================
# Linear algebra utilities
# =============================================================================
def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def chol_inv(A: np.ndarray, jitter: float = 1e-9, max_tries: int = 4) -> np.ndarray:
    A = _sym(A)
    n = A.shape[0]; I = np.eye(n); eps = jitter
    for _ in range(max_tries):
        try:
            L = np.linalg.cholesky(A + eps * I)
            X = np.linalg.solve(L, I)
            return np.linalg.solve(L.T, X)
        except np.linalg.LinAlgError:
            eps *= 10.0
    return np.linalg.inv(A + eps * I)

def chol_solve(A: np.ndarray, B: np.ndarray, jitter: float = 1e-9, max_tries: int = 4) -> np.ndarray:
    A = _sym(A)
    eps = jitter
    for _ in range(max_tries):
        try:
            L = np.linalg.cholesky(A + eps * np.eye(A.shape[0]))
            Y = np.linalg.solve(L, B)
            return np.linalg.solve(L.T, Y)
        except np.linalg.LinAlgError:
            eps *= 10.0
    return np.linalg.lstsq(A, B, rcond=None)[0]

# =============================================================================
# Angle wrapping
# =============================================================================
def angle_normalize(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def wrap_error(e: np.ndarray, wrap_idx: Optional[List[int]] = None) -> np.ndarray:
    if not wrap_idx:
        return e
    e = e.copy()
    for i in wrap_idx:
        e[i] = angle_normalize(e[i])
    return e

# =============================================================================
# Finite-diff Jacobian w.r.t x
# =============================================================================
def jacobian_x_fd(F, x, u, eps: float = 1e-6) -> np.ndarray:
    n = x.size
    J = np.zeros((n, n))
    f0 = F(x, u)
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        J[:, i] = (F(x + dx, u) - f0) / eps
    return J

# =============================================================================
# Build a feasible negative-time prefix by Newton pre-image
# =============================================================================
def newton_preimage_step(F, x_next, u_prev, max_iter=10, tol=1e-9) -> np.ndarray:
    """Solve F(x_prev, u_prev) = x_next for x_prev via Newton; start from x_prev=x_next."""
    x = x_next.copy()
    for _ in range(max_iter):
        fx = F(x, u_prev)
        g  = fx - x_next
        if np.linalg.norm(g) < tol:
            return x
        J = jacobian_x_fd(F, x, u_prev)
        # solve J * dx = g  => x <- x - dx
        try:
            dx = np.linalg.solve(J, g)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(J, g, rcond=None)[0]
        # damping to avoid overshoot
        step = 1.0
        x_new = x - step * dx
        # simple backtracking if not improving residual
        if np.linalg.norm(F(x_new, u_prev) - x_next) > np.linalg.norm(g):
            step = 0.5
            x_new = x - step * dx
        x = x_new
    return x  # return best we have

def extend_nominal_backward(F, X, U, u_ref, S_back: int) -> (np.ndarray, np.ndarray):
    """
    Build a negative-time prefix of length S_back by solving F(x_{-1}, u_ref)=x_0, F(x_{-2},u_ref)=x_{-1}, ...
    Returns X_ext (len = S_back + len(X)), U_ext (len = S_back + len(U)).
    """
    if S_back <= 0:
        return X.copy(), U.copy()
    n = X.shape[1]; m = U.shape[1]
    X_ext = np.zeros((X.shape[0] + S_back, n))
    U_ext = np.zeros((U.shape[0] + S_back, m))
    # put original at the tail
    X_ext[S_back:] = X
    U_ext[S_back:] = U
    # build prefix
    x_curr = X[0].copy()
    for s in range(1, S_back + 1):
        x_prev = newton_preimage_step(F, x_curr, u_ref)
        X_ext[S_back - s] = x_prev
        U_ext[S_back - s] = u_ref
        x_curr = x_prev
    return X_ext, U_ext

# =============================================================================
# Linearization along a nominal trajectory
# =============================================================================
def linearize_central_diff_traj(F, X, U, epsx=1e-5, epsu=1e-5, relx=1e-6, relu=1e-6):
    N = len(U); n = X.shape[1]; m = U.shape[1]
    A_list, B_list = [], []
    for k in range(N):
        x = X[k].copy(); u = U[k].copy()
        A = np.zeros((n, n)); B = np.zeros((n, m))
        for i in range(n):
            hi = max(epsx, relx * max(1.0, abs(x[i])))
            xp = x.copy(); xm = x.copy()
            xp[i] += hi;   xm[i] -= hi
            A[:, i] = (F(xp, u) - F(xm, u)) / (2.0 * hi)
        for j in range(m):
            hj = max(epsu, relu * max(1.0, abs(u[j])))
            up = u.copy(); um = u.copy()
            up[j] += hj;   um[j] -= hj
            B[:, j] = (F(x, up) - F(x, um)) / (2.0 * hj)
        A_list.append(A); B_list.append(B)
    return A_list, B_list

def linearize_forward_diff_traj(F, X, U, epsx=1e-6, epsu=1e-6):
    N = len(U); n = X.shape[1]; m = U.shape[1]
    A_list, B_list = [], []
    I_n, I_m = np.eye(n), np.eye(m)
    for k in range(N):
        x = X[k]; u = U[k]; f0 = F(x, u)
        A = np.zeros((n, n)); B = np.zeros((n, m))
        for i in range(n): A[:, i] = (F(x + epsx * I_n[i], u) - f0) / epsx
        for j in range(m): B[:, j] = (F(x, u + epsu * I_m[j]) - f0) / epsu
        A_list.append(A); B_list.append(B)
    return A_list, B_list

# =============================================================================
# Affine residuals for linearized dynamics
# =============================================================================
def compute_affine_residuals(F, X, U):
    return [(F(X[k], U[k]) - X[k + 1]).reshape(-1, 1) for k in range(len(U))]

# =============================================================================
# Augmented blocks (Propagator) — unchanged
# =============================================================================
def build_augmented_sequence_QR(
    F, A_list, B_list, X, U, xg, u_ref, Q, R, w,
    wrap_idx: Optional[List[int]] = None,
    q_reg: float = 1e-9,
    rho_reg: float = 1e-12,
):
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
        Qk[:n,  n] = (Q @ e).ravel()
        Qk[ n, :n] = (Q @ e).ravel()
        Qk[ n,  n] = float(e.T @ Q @ e) + 2.0 * float(w) + rho_reg
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
    X, xg, alpha, Qtilde, wrap_idx: Optional[List[int]] = None, rho_reg=1e-12
):
    N = X.shape[0] - 1; QT = []; P = _sym(alpha * Qtilde)
    for t in range(1, N + 1):
        e  = wrap_error((X[t] - xg), wrap_idx).reshape(-1, 1)
        px = P @ e; p0 = 0.5 * float((e.T @ (P @ e)).item())
        Qt = np.zeros((P.shape[0] + 1, P.shape[1] + 1), dtype=float)
        Qt[:-1, :-1] = P
        Qt[:-1, -1]  = px.ravel()
        Qt[-1,  :-1] = px.ravel()
        Qt[-1,  -1]  = 2.0 * p0 + rho_reg
        QT.append(_sym(Qt))
    return QT

def propagator_all_Jt_aug(A_aug, B_aug, Q_aug, R_list, z0, QT_aug_list, T_use=None, R_inv_cached: Optional[np.ndarray] = None):
    N = len(A_aug) if T_use is None else T_use
    Rinvs = ([chol_inv(R_list[k]) for k in range(N)] if R_inv_cached is None else [R_inv_cached for _ in range(N)])
    E_list, F_list, G_list = [], [], []
    for k in range(N):
        Ek = chol_inv(Q_aug[k]); Fk = Ek @ A_aug[k].T
        Gk = A_aug[k] @ Ek @ A_aug[k].T + B_aug[k] @ Rinvs[k] @ B_aug[k].T
        E_list.append(Ek); F_list.append(Fk); G_list.append(_sym(Gk))
    Ebar = [E_list[0].copy()]; Fbar = [F_list[0].copy()]; Gbar = [G_list[0].copy()]
    for k in range(1, N):
        Ek, Fk, Gk = E_list[k], F_list[k], G_list[k]
        W = chol_inv(Ek + Gbar[-1])
        Ebar.append(_sym(Ebar[-1] - Fbar[-1] @ W @ Fbar[-1].T))
        Fbar.append(Fbar[-1] @ W @ Fk)
        Gbar.append(_sym(Gk - Fk.T @ W @ Fk))
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
# Backward pass (standard iLQR, no 2nd-order dynamics)
# =============================================================================
def value_expansions_and_gains_prefix(
    A_list, B_list, X, U, xg, u_ref, Q, R, alpha, T_bar, S_right,
    lm_lambda: float = 1e-6, w_stage: float = 0.0, wrap_idx: Optional[List[int]] = None
):
    """
    Compute Vxx,Vx,V0 and K,k for all t in [-S_right .. T_bar] using a single backward pass with terminal at T_bar.
    Arrays are returned with an offset: index i corresponds to t = i - S_right.
    """
    n, m = X.shape[1], U.shape[1]
    L = T_bar + S_right  # max index for gains arrays
    Vxx = [np.zeros((n, n)) for _ in range(L + 1)]
    Vx  = [np.zeros(n)       for _ in range(L + 1)]
    V0  = [0.0               for _ in range(L + 1)]
    K   = [None] * (L)  # defined for t <= T_bar-1 (shifted)
    k   = [None] * (L)

    # terminal at t = T_bar  => index iT = T_bar + S_right
    iT = T_bar + S_right
    eT = np.atleast_1d(wrap_error(X[iT] - xg, wrap_idx)).reshape(-1)
    Vxx[iT] = _sym(alpha * np.eye(n))
    Vx[iT]  = alpha * eT
    V0[iT]  = 0.5 * alpha * float(eT @ eT)

    # backward over t = T_bar-1 down to -S_right
    for t in reversed(range(-S_right, T_bar)):
        i = t + S_right
        e  = np.atleast_1d(wrap_error(X[i] - xg, wrap_idx)).reshape(-1)
        du = np.atleast_1d(U[i] - u_ref).reshape(-1)

        lx, lu = Q @ e, R @ du
        l0 = 0.5 * float(e @ (Q @ e)) + 0.5 * float(du @ (R @ du)) + float(w_stage)

        A, B = A_list[i], B_list[i]
        Qx   = lx + A.T @ Vx[i + 1]
        Qu   = lu + B.T @ Vx[i + 1]
        Qxx  = Q  + A.T @ Vxx[i + 1] @ A
        Quu  = R  + B.T @ Vxx[i + 1] @ B
        Qux  = B.T @ Vxx[i + 1] @ A

        Quu_reg  = _sym(Quu) + lm_lambda * np.eye(m)
        invQuuQu  = chol_solve(Quu_reg, Qu)
        invQuuQux = chol_solve(Quu_reg, Qux)

        k[i] = -invQuuQu
        K[i] = -invQuuQux

        Vxx[i] = _sym(Qxx - Qux.T @ invQuuQux)
        Vx[i]  = Qx  - Qux.T @ invQuuQu
        V0[i]  = l0  + V0[i + 1] - 0.5 * float(Qu.T @ invQuuQu)

    return Vxx, Vx, V0, K, k  # with offset S_right

# =============================================================================
# One-pass: pick T and build the plot window (single backward)
# =============================================================================
def onepass_pick_T_singlepass(
    Vxx, Vx, V0, X_ext, x0, T_bar, T_min, T_max, S_left, S_right, wrap_idx
):
    """
    Use the *single* backward expansion around T̄ to evaluate the window.
    Returns: best T_star and a J_window (NaN outside [T̄-S_L, T̄+S_R]).
    """
    Nmax = T_max
    Jw = np.full(Nmax, np.nan, dtype=float)
    bestJ, bestT = np.inf, T_bar
    # window clamps
    L = max(T_min, T_bar - S_left)
    R = min(T_max, T_bar + S_right)
    for T in range(L, R + 1):
        t0 = T_bar - T          # may be negative
        i  = t0 + S_right       # shift to array index
        dx0 = wrap_error((x0 - X_ext[i]).reshape(-1), wrap_idx)
        JT = 0.5 * float(dx0 @ (Vxx[i] @ dx0)) + float(Vx[i] @ dx0) + float(V0[i])
        Jw[T - 1] = JT
        if JT < bestJ:
            bestJ, bestT = JT, T
    return bestT, Jw

# =============================================================================
# One-pass rollout with negative-time gains available
# =============================================================================
def onepass_rollout_singlepass(
    F, X_ext, U_ext, xg, u_ref, Q, R, alpha, w, K, k, T_bar, T_star,
    alphas=(1.0, 0.5, 0.25, 0.1, 0.05), wrap_idx=None, S_right=0
):
    """
    Use gains at indices t0..t0+T_star-1, where t0 = T_bar - T_star (may be negative).
    Arrays K,k,X_ext,U_ext are indexed with shift S_right.
    """
    t0 = T_bar - T_star
    off = S_right
    N = U_ext.shape[0] - off  # rollout length limited by original horizon
    J_best, X_best, U_best = None, None, None

    for a in alphas:
        Xn = np.zeros((X_ext.shape[0] - off, X_ext.shape[1]))  # start from t=0
        Un = U_ext[off:].copy()
        Xn[0] = X_ext[off]  # same initial state

        for t in range(T_star):
            idx = t0 + t + off
            dx = wrap_error((Xn[t] - X_ext[idx]).reshape(-1), wrap_idx)
            du = (K[idx] @ dx + a * k[idx]).reshape(-1)
            Un[t] = (U_ext[idx] + du).reshape(-1)
            Xn[t + 1] = F(Xn[t], Un[t])

        for t in range(T_star, Un.shape[0]):
            Xn[t + 1] = F(Xn[t], Un[t])

        J_new = cost_timeopt_true(Xn, Un, xg, u_ref, Q, R, alpha, w, T_star, wrap_idx)
        if J_best is None or J_new < J_best:
            J_best, X_best, U_best = J_new, Xn, Un

    return J_best, X_best, U_best

# =============================================================================
# Rollout & true cost
# =============================================================================
def rollout(F, x0, U):
    N = len(U); n = len(x0)
    X = np.zeros((N + 1, n)); X[0] = x0
    for k in range(N):
        X[k + 1] = F(X[k], U[k])
    return X

def cost_timeopt_true(X, U, xg, u_ref, Q, R, alpha, w, T_star, wrap_idx=None):
    c = 0.0
    for k in range(T_star):
        e  = np.atleast_1d(wrap_error(X[k] - xg, wrap_idx)).reshape(-1)
        du = np.atleast_1d(U[k] - u_ref).reshape(-1)
        c += 0.5 * float(e @ (Q @ e)) + 0.5 * float(du @ (R @ du)) + float(w)
    eT = np.atleast_1d(wrap_error(X[T_star] - xg, wrap_idx)).reshape(-1)
    c += 0.5 * float(alpha * (eT @ eT))
    return float(c)

# =============================================================================
# Fixed-horizon iLQR (用于 propagator/bruteforce 分支)
# =============================================================================
def backward_pass_truncated(A_list, B_list, X, U, xg, u_ref, Q, R, alpha, T_star, lm_lambda=1e-3, wrap_idx=None):
    n, m = X.shape[1], U.shape[1]
    k_list = [None] * T_star; K_list = [None] * T_star
    eT = np.atleast_1d(wrap_error(X[T_star] - xg, wrap_idx)).reshape(-1)
    Vx  = alpha * eT; Vxx = _sym(alpha * np.eye(n))
    for k in reversed(range(T_star)):
        e  = np.atleast_1d(wrap_error(X[k] - xg, wrap_idx)).reshape(-1)
        du = np.atleast_1d(U[k] - u_ref).reshape(-1)
        lx, lu = Q @ e, R @ du
        A, B = A_list[k], B_list[k]
        Qx  = lx + A.T @ Vx
        Qu  = lu + B.T @ Vx
        Qxx = Q  + A.T @ Vxx @ A
        Quu = R  + B.T @ Vxx @ B
        Qux = B.T @ Vxx @ A
        Quu_reg = _sym(Quu) + lm_lambda * np.eye(m)
        try:
            np.linalg.cholesky(Quu_reg)
        except np.linalg.LinAlgError:
            return None, None, False
        kappa = -chol_solve(Quu_reg, Qu)
        Kk    = -chol_solve(Quu_reg, Qux)
        k_list[k] = kappa; K_list[k] = Kk
        Vx  = Qx  + Kk.T @ Qu  + Qux.T @ kappa + Kk.T @ Quu @ kappa
        Vxx = _sym(Qxx + Kk.T @ Qux + Qux.T @ Kk + Kk.T @ Quu @ Kk)
    return k_list, K_list, True

def forward_linesearch_fixedT(
    F, X, U, xg, u_ref, Q, R, alpha, w, T_star,
    k_list, K_list, alphas=(1.0, 0.5, 0.25, 0.1, 0.05), wrap_idx=None
):
    J_old = cost_timeopt_true(X, U, xg, u_ref, Q, R, alpha, w, T_star, wrap_idx)
    N = len(U)
    for a in alphas:
        U_new = U.copy()
        X_new = np.zeros_like(X); X_new[0] = X[0]
        for k in range(T_star):
            dx = wrap_error((X_new[k] - X[k]).reshape(-1), wrap_idx)
            du = (K_list[k] @ dx + a * k_list[k]).reshape(-1)
            U_new[k] = (U[k] + du).reshape(-1)
            X_new[k + 1] = F(X_new[k], U_new[k])
        for k in range(T_star, N):
            X_new[k + 1] = F(X_new[k], U_new[k])
        J_new = cost_timeopt_true(X_new, U_new, xg, u_ref, Q, R, alpha, w, T_star, wrap_idx)
        if J_new < J_old:
            return X_new, U_new, J_new, True
    return X, U, J_old, False

# =============================================================================
# Bruteforce J_t via backward expansion
# =============================================================================
def bruteforce_all_Jt_backward_expansion(A_list, B_list, X, U, xg, u_ref, Q, R, alpha, w, T_max, lm_lambda=1e-6, wrap_idx=None):
    J = np.zeros(T_max)
    for T in range(1, T_max + 1):
        # 一次 backward，取 V0[0]
        n, m = X.shape[1], U.shape[1]
        Vxx = [np.zeros((n, n)) for _ in range(T + 1)]
        Vx  = [np.zeros(n)       for _ in range(T + 1)]
        V0  = [0.0               for _ in range(T + 1)]
        eT = np.atleast_1d(wrap_error(X[T] - xg, wrap_idx)).reshape(-1)
        Vxx[T] = alpha * np.eye(n); Vx[T] = alpha * eT; V0[T] = 0.5 * alpha * float(eT @ eT)
        for t in reversed(range(T)):
            e  = np.atleast_1d(wrap_error(X[t] - xg, wrap_idx)).reshape(-1)
            du = np.atleast_1d(U[t] - u_ref).reshape(-1)
            lx, lu = Q @ e, R @ du
            l0 = 0.5 * float(e @ (Q @ e)) + 0.5 * float(du @ (R @ du)) + float(w)
            A, B = A_list[t], B_list[t]
            Qx   = lx + A.T @ Vx[t + 1]
            Qu   = lu + B.T @ Vx[t + 1]
            Qxx  = Q  + A.T @ Vxx[t + 1] @ A
            Quu  = R  + B.T @ Vxx[t + 1] @ B
            Qux  = B.T @ Vxx[t + 1] @ A
            Quu_reg = _sym(Quu) + lm_lambda * np.eye(m)
            invQuuQu  = chol_solve(Quu_reg, Qu)
            invQuuQux = chol_solve(Quu_reg, Qux)
            Vxx[t] = _sym(Qxx - Qux.T @ invQuuQux)
            Vx[t]  = Qx  - Qux.T @ invQuuQu
            V0[t]  = l0  + V0[t + 1] - 0.5 * float(Qu.T @ invQuuQu)
        J[T - 1] = float(V0[0])
    return J

# =============================================================================
# Top-level outer loop
# =============================================================================
def ilqr_timeopt(
    F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max,
    method: str = "propagator",
    max_iter: int = 15,
    lm_init: float = 1e-3,
    S_window: int = 20,
    S_left: Optional[int] = None,
    S_right: Optional[int] = None,
    wrap_idx: Optional[List[int]] = None,
    use_central_diff: bool = True,
):
    assert method in ("propagator", "onepass", "bruteforce")

    # 初始名义轨迹
    U = np.tile(u_ref, (N, 1))
    X = rollout(F, x0, U)

    J_hist, T_hist = [], []
    timers = {"linearize": 0.0, "select": 0.0, "backward": 0.0, "forward": 0.0}
    lm = lm_init

    # 初次线性化
    t0 = time.perf_counter()
    if use_central_diff:
        A_list, B_list = linearize_central_diff_traj(F, X, U)
    else:
        A_list, B_list = linearize_forward_diff_traj(F, X, U)
    timers["linearize"] += time.perf_counter() - t0

    # 初始 T̄
    if method == "propagator":
        t_sel0 = time.perf_counter()
        A_aug, B_aug, Q_aug, R_list, z0, R_inv = build_augmented_sequence_QR(
            F, A_list, B_list, X, U, xg, u_ref, Q, R, w, wrap_idx=wrap_idx
        )
        QT_list0 = build_terminal_aug_list(X, xg, alpha, np.eye(X.shape[1]), wrap_idx=wrap_idx)
        J_prop = propagator_all_Jt_aug(A_aug, B_aug, Q_aug, R_list, z0, QT_list0, T_use=T_max, R_inv_cached=R_inv)
        T_bar = int(np.argmin(J_prop[T_min - 1 : T_max]) + T_min)
        timers["select"] += time.perf_counter() - t_sel0
    elif method == "bruteforce":
        t_sel0 = time.perf_counter()
        J_back = bruteforce_all_Jt_backward_expansion(A_list, B_list, X, U, xg, u_ref, Q, R, alpha, w, T_max, wrap_idx=wrap_idx)
        T_bar = int(np.argmin(J_back[T_min - 1 : T_max]) + T_min)
        timers["select"] += time.perf_counter() - t_sel0
    else:
        T_bar = (T_min + T_max) // 2  # onepass: 与论文一致，不借助别的方法

    # 先做一次固定 T̄ 的更新热身
    t2 = time.perf_counter()
    k_list, K_list, ok = backward_pass_truncated(A_list, B_list, X, U, xg, u_ref, Q, R, alpha, T_bar, lm_lambda=lm, wrap_idx=wrap_idx)
    timers["backward"] += time.perf_counter() - t2
    if ok:
        t3 = time.perf_counter()
        X, U, J0, _ = forward_linesearch_fixedT(F, X, U, xg, u_ref, Q, R, alpha, w, T_bar, k_list, K_list, wrap_idx=wrap_idx)
        timers["forward"] += time.perf_counter() - t3
        J_hist.append(J0); T_hist.append(T_bar)

    # 窗口大小
    S_left  = S_window if S_left  is None else S_left
    S_right = S_window if S_right is None else S_right
    last_window_curve = None  # 用于绘图（只画最后一次窗口）

    # 外层迭代
    for _ in range(max_iter):
        # 线性化（必要时先做负时间前缀）
        t0 = time.perf_counter()
        if method == "onepass":
            # 构造负时间可行前缀（论文 IV‑B）:contentReference[oaicite:11]{index=11}】
            X_ext, U_ext = extend_nominal_backward(F, X, U, u_ref, S_back=S_right)
            if use_central_diff:
                A_ext, B_ext = linearize_central_diff_traj(F, X_ext, U_ext)
            else:
                A_ext, B_ext = linearize_forward_diff_traj(F, X_ext, U_ext)
        else:
            X_ext, U_ext = X, U
            if use_central_diff:
                A_ext, B_ext = linearize_central_diff_traj(F, X_ext, U_ext)
            else:
                A_ext, B_ext = linearize_forward_diff_traj(F, X_ext, U_ext)
        timers["linearize"] += time.perf_counter() - t0

        if method == "propagator":
            t1 = time.perf_counter()
            A_aug, B_aug, Q_aug, R_list, z0, R_inv = build_augmented_sequence_QR(
                F, A_ext, B_ext, X_ext, U_ext, xg, u_ref, Q, R, w, wrap_idx=wrap_idx
            )
            QT_list = build_terminal_aug_list(X_ext, xg, alpha, np.eye(X.shape[1]), wrap_idx=wrap_idx)
            J_all = propagator_all_Jt_aug(A_aug, B_aug, Q_aug, R_list, z0, QT_list, T_use=T_max, R_inv_cached=R_inv)
            T_star = int(np.argmin(J_all[T_min - 1 : T_max]) + T_min)
            timers["select"] += time.perf_counter() - t1

            t2 = time.perf_counter()
            k_list, K_list, ok = backward_pass_truncated(A_ext, B_ext, X_ext, U_ext, xg, u_ref, Q, R, alpha, T_star, lm_lambda=lm, wrap_idx=wrap_idx)
            timers["backward"] += time.perf_counter() - t2
            if not ok:
                lm *= 10.0
                continue
            t3 = time.perf_counter()
            Xn, Un, Jn, acc = forward_linesearch_fixedT(F, X, U, xg, u_ref, Q, R, alpha, w, T_star, k_list, K_list, wrap_idx=wrap_idx)
            timers["forward"] += time.perf_counter() - t3

        elif method == "onepass":
            # 单次 backward at T̄
            t1 = time.perf_counter()
            Vxx, Vx, V0, Kp, kp = value_expansions_and_gains_prefix(
                A_ext, B_ext, X_ext, U_ext, xg, u_ref, Q, R, alpha, T_bar,
                S_right=S_right, lm_lambda=lm, w_stage=w, wrap_idx=wrap_idx
            )
            # 选择窗口（单窗口）
            T_star, J_win = onepass_pick_T_singlepass(
                Vxx, Vx, V0, X_ext, X_ext[S_right], T_bar, T_min, T_max, S_left, S_right, wrap_idx
            )
            timers["select"] += time.perf_counter() - t1
            last_window_curve = J_win

            # 根据论文：若代价不降，先试更小 α；再不行缩 S 重试:contentReference[oaicite:13]{index=13}】
            t3 = time.perf_counter()
            acc = False
            S_L_cur, S_R_cur = S_left, S_right
            J_prev = J_hist[-1] if len(J_hist) else np.inf

            for _shrink in range(3):  # 最多缩 3 次窗口
                Jn, Xn, Un = onepass_rollout_singlepass(
                    F, X_ext, U_ext, xg, u_ref, Q, R, alpha, w,
                    Kp, kp, T_bar, T_star, wrap_idx=wrap_idx, S_right=S_right
                )
                if Jn < J_prev:
                    acc = True
                    break
                # 缩小窗口并重选 T
                S_L_cur = max(1, S_L_cur // 2)
                S_R_cur = max(1, S_R_cur // 2)
                T_star, J_win = onepass_pick_T_singlepass(
                    Vxx, Vx, V0, X_ext, X_ext[S_right], T_bar, T_min, T_max, S_L_cur, S_R_cur, wrap_idx
                )
                last_window_curve = J_win

            timers["forward"] += time.perf_counter() - t3

        else:  # 'bruteforce'
            t1 = time.perf_counter()
            J_all = bruteforce_all_Jt_backward_expansion(A_ext, B_ext, X_ext, U_ext, xg, u_ref, Q, R, alpha, w, T_max, wrap_idx=wrap_idx)
            T_star = int(np.argmin(J_all[T_min - 1 : T_max]) + T_min)
            timers["select"] += time.perf_counter() - t1

            t2 = time.perf_counter()
            k_list, K_list, ok = backward_pass_truncated(A_ext, B_ext, X_ext, U_ext, xg, u_ref, Q, R, alpha, T_star, lm_lambda=lm, wrap_idx=wrap_idx)
            timers["backward"] += time.perf_counter() - t2
            if not ok:
                lm *= 10.0
                continue

            t3 = time.perf_counter()
            Xn, Un, Jn, acc = forward_linesearch_fixedT(F, X_ext, U_ext, xg, u_ref, Q, R, alpha, w, T_star, k_list, K_list, wrap_idx=wrap_idx)
            timers["forward"] += time.perf_counter() - t3

        # 接受/拒绝 + LM
        if acc:
            # 注意：onepass 的 Xn,Un, 以 t=0 开始（去掉负前缀）
            X = Xn; U = Un; T_bar = T_star
            J_hist.append(Jn); T_hist.append(T_star)
            lm = max(lm / 10.0, 1e-12)
        else:
            lm *= 10.0

        # 终止条件
        if len(J_hist) >= 2:
            rel = abs(J_hist[-1] - J_hist[-2]) / (abs(J_hist[-2]) + 1e-12)
            if rel < 1e-4 and (len(T_hist) >= 3 and len(set(T_hist[-3:])) == 1):
                break

    # ====== 结果曲线（仅用于可视化与一致性检查）======
    if use_central_diff:
        A_list, B_list = linearize_central_diff_traj(F, X, U)
    else:
        A_list, B_list = linearize_forward_diff_traj(F, X, U)

    J_back = bruteforce_all_Jt_backward_expansion(A_list, B_list, X, U, xg, u_ref, Q, R, alpha, w, T_max, wrap_idx=wrap_idx)
    A_aug, B_aug, Q_aug, R_list, z0, R_inv = build_augmented_sequence_QR(
        F, A_list, B_list, X, U, xg, u_ref, Q, R, w, wrap_idx=wrap_idx
    )
    QT_list = build_terminal_aug_list(X, xg, alpha, np.eye(X.shape[1]), wrap_idx=wrap_idx)
    J_prop = propagator_all_Jt_aug(A_aug, B_aug, Q_aug, R_list, z0, QT_list, T_use=T_max, R_inv_cached=R_inv)
    diff = J_prop - J_back
    consistency = {"max_abs_diff": float(np.max(np.abs(diff))), "rmse": float(np.sqrt(np.mean(diff**2)))}

    if method == "propagator":
        J_curve = J_prop
    elif method == "onepass":
        J_curve = last_window_curve if last_window_curve is not None else np.full(T_max, np.nan)
    else:
        J_curve = J_back

    return {
        "X": X, "U": U,
        "J_hist": J_hist, "T_hist": T_hist,
        "timers": timers, "J_curve": J_curve,
        "T_star": (T_hist[-1] if len(T_hist) > 0 else T_bar),
        "consistency_check": consistency,
    }

# =============================================================================
# Example systems (与你原来一致)
# =============================================================================
def make_double_integrator(dt=0.05, N=120):
    def F(x, u): return np.array([x[0] + dt * x[1], x[1] + dt * u[0]])
    x0 = np.array([1.0, 0.0]); xg = np.array([2.0, 0.0]); u_ref = np.array([0.0])
    Q = np.diag([1.0, 0.1]); R = np.array([[1e-2]])
    alpha = 50.0; w = 0.02; T_min, T_max = 10, 80; wrap_idx = []
    return F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max, wrap_idx

def make_segway(dt=0.02, N=240):
    g = 9.81; r = 0.15; M = 1.0; m = 2.0; l = 0.5; I = (1.0 / 3.0) * m * l * l
    a1 = M + m; a2 = m * l; a3 = I + m * l * l; Den = a1 * a3 - a2 * a2
    A_tau = a3 / (r * Den) - a2 / Den; A_th  = -(a2 * m * g * l) / Den
    B_tau = -a2 / (r * Den) + a1 / Den; B_th  = (a1 * m * g * l) / Den
    def F(x, u):
        x_pos, x_dot, th, th_dot = x; tau = u[0]
        xdd  = A_tau * tau + A_th * th; thdd = B_tau * tau + B_th * th
        return np.array([x_pos + dt * x_dot, x_dot + dt * xdd, angle_normalize(th + dt * th_dot), th_dot + dt * thdd])
    x0 = np.array([2.0, 0.0, 2.0, 0.0]); xg = np.array([0.0, 0.0, 0.0, 0.0]); u_ref = np.array([0.0])
    Q = np.diag([0.1, 0.02, 200.0, 4.0]).astype(float); R = np.array([[0.05]]).astype(float)
    alpha = 120.0; w = 2e-2
    T_min, T_max = 40, 200; wrap_idx = [2]
    return F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max, wrap_idx

def make_ballbot(dt=0.02, N=260):
    g = 9.81; r = 0.12; m_ball = 1.2; I_ball = (2.0 / 5.0) * m_ball * r * r
    M_eff = m_ball + I_ball / (r * r); m_body = 2.0; l = 0.55
    def F(x, u):
        x_pos, x_dot, th, th_dot = x; tau = u[0]; force = tau / r
        s = np.sin(th); c = np.cos(th); total_mass = M_eff + m_body; polemass_length = m_body * l
        temp = (force + polemass_length * th_dot ** 2 * s) / total_mass
        th_acc = (g * s - c * temp) / (l * (4.0 / 3.0 - m_body * c * c / total_mass))
        x_acc  = temp - polemass_length * th_acc * c / total_mass
        return np.array([x_pos + dt * x_dot, x_dot + dt * x_acc, angle_normalize(th + dt * th_dot), th_dot + dt * th_acc])
    x0 = np.array([0.05, 0.0, 0.08, 0.0]); xg = np.array([0.0, 0.0, 0.0, 0.0]); u_ref = np.array([0.0])
    Q = np.diag([1.0, 0.1, 25.0, 1.0]).astype(float); R = np.array([[0.25]]).astype(float)
    alpha = 220.0; w = 1e-4; T_min, T_max = 60, 200; wrap_idx = [2]
    return F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max, wrap_idx

def make_quadrotor(dt=0.05, N=160):
    m, g = 1.0, 9.81; Ix, Iy, Iz = 0.02, 0.02, 0.04; kv, kw = 0.05, 0.01
    def rotm(phi, th, psi):
        s, c = np.sin, np.cos
        Rz = np.array([[c(psi), -s(psi), 0], [s(psi), c(psi), 0], [0, 0, 1]])
        Ry = np.array([[c(th), 0, s(th)], [0, 1, 0], [-s(th), 0, c(th)]])
        Rx = np.array([[1, 0, 0], [0, c(phi), -s(phi)], [0, s(phi), c(phi)]])
        return Rz @ Ry @ Rx
    def Tmat(phi, th):
        s, c, t = np.sin, np.cos, np.tan; sec = lambda x: 1.0 / np.cos(x)
        return np.array([[1, s(phi) * t(th), c(phi) * t(th)],[0, c(phi), -s(phi)],[0, s(phi) * sec(th), c(phi) * sec(th)]])
    def F(x, u):
        pos = x[0:3]; vel = x[3:6]; phi, th, psi = x[6:9]; omg = x[9:12]
        thrust = u[0]; tau = u[1:4]; Rb = rotm(phi, th, psi); e3 = np.array([0, 0, 1.0])
        acc = (Rb @ (e3 * thrust)) / m - np.array([0, 0, g]) - kv * vel
        eulerdot = Tmat(phi, th) @ omg
        I = np.diag([Ix, Iy, Iz]); omgdot = np.linalg.inv(I) @ (tau - np.cross(omg, I @ omg)) - kw * omg
        xdot = np.zeros(12); xdot[0:3] = vel; xdot[3:6] = acc; xdot[6:9] = eulerdot; xdot[9:12] = omgdot
        return x + dt * xdot
    x0 = np.zeros(12); x0[0:3] = [2.0, 2.0, 2.0]; xg = np.zeros(12); u_ref = np.array([m * g, 0.0, 0.0, 0.0])
    Q = np.diag([5,5,5,1,1,1,20,20,10,1,1,1]).astype(float); R = np.diag([1e-3,1e-2,1e-2,1e-2]).astype(float)
    alpha = 300.0; w = 0.005; T_min, T_max = 40, 160; wrap_idx = [6,7,8]
    return F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max, wrap_idx

# =============================================================================
# Plotting & main (与原版一致)
# =============================================================================
def plot_trajectories(results, methods, case_name, output_dir, state_names=None, control_names=None):
    import os
    n_states = results[methods[0]]["X"].shape[1]
    n_controls = results[methods[0]]["U"].shape[1]
    if state_names is None: state_names = [f"x_{i+1}" for i in range(n_states)]
    if control_names is None: control_names = [f"u_{i+1}" for i in range(n_controls)]
    styles = {
        "propagator": {"color": "#2E86AB", "linestyle": "-",  "linewidth": 2.5, "alpha": 0.85},
        "onepass":    {"color": "#A23B72", "linestyle": "--", "linewidth": 2.5, "alpha": 0.85},
        "bruteforce": {"color": "#F18F01", "linestyle": "-.", "linewidth": 2.0, "alpha": 0.75},
    }
    n_rows = max(n_states, n_controls)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows))
    for i in range(n_states):
        ax = axes[i, 0]
        for m in methods:
            X = results[m]["X"]; T = results[m]["T_star"]
            ax.plot(np.arange(T + 1), X[: T + 1, i], label=m.capitalize(), **styles[m])
        ax.set_xlabel("t"); ax.set_ylabel(state_names[i]); ax.grid(True, alpha=0.3)
    for j in range(n_controls):
        ax = axes[j, 1]
        for m in methods:
            U = results[m]["U"]; T = results[m]["T_star"]
            ax.step(np.arange(T), U[:T, j], where="post", label=m.capitalize(), **styles[m])
        ax.set_xlabel("t"); ax.set_ylabel(control_names[j]); ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=10)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = f"{output_dir}/{case_name}_trajectories.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out

def main():
    import os, pandas as pd, numpy as np
    output_dir = "ilqr_propagator_results"; os.makedirs(output_dir, exist_ok=True)
    cases = [
        ("DoubleIntegrator", make_double_integrator, ["x", "v"], ["u"]),
        ("Segway_Balance",   make_segway,           ["x", "ẋ", "θ", "θ̇"], ["τ"]),
        ("Ballbot_Balance",  make_ballbot,          ["x", "ẋ", "θ", "θ̇"], ["τ"]),
        ("Quadrotor_Hover",  make_quadrotor,        ["x","y","z","vx","vy","vz","φ","θ","ψ","ωx","ωy","ωz"], ["Thrust","τx","τy","τz"]),
    ]
    methods = ["propagator", "onepass", "bruteforce"]
    summary_rows = []
    for case_name, maker, state_names, control_names in cases:
        print(f"\n{'='*60}\nRunning: {case_name}\n{'='*60}")
        F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max, wrap_idx = maker()
        results = {}
        for method in methods:
            print(f"\n  Method: {method}")
            res = ilqr_timeopt(
                F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max,
                method=method, max_iter=20, lm_init=1e-3,
                S_window=10, wrap_idx=wrap_idx, use_central_diff=True,
            )
            results[method] = res
            T_star = res["T_star"]
            J_final = res["J_hist"][-1] if len(res["J_hist"]) > 0 else np.nan
            timers = res["timers"]; total_time = sum(timers.values())
            cc = res.get("consistency_check", {"max_abs_diff": np.nan, "rmse": np.nan})
            print(f"    T* = {T_star}, J* = {J_final:.4f}")
            print(f"    Total time: {total_time:.4f}s")
            print(f"    Breakdown: linearize={timers['linearize']:.4f}s, select={timers['select']:.4f}s, backward={timers['backward']:.4f}s, forward={timers['forward']:.4f}s")
            x_initial = res["X"][0]; x_final = res["X"][T_star]
            print(f"    Initial state: [{', '.join([f'{x:.4f}' for x in x_initial])}]")
            print(f"    Final state:   [{', '.join([f'{x:.4f}' for x in x_final])}]")
            err = np.linalg.norm(wrap_error(x_final - xg, wrap_idx))
            print(f"    Final state error: {err:.6f}")
            print(f"    Consistency (prop vs back) max|Δ| = {cc['max_abs_diff']:.3e}, rmse = {cc['rmse']:.3e}")
            summary_rows.append({
                "case": case_name, "method": method, "T_star": T_star, "J_star": J_final, "total_time": total_time,
                "t_linearize": timers["linearize"], "t_select": timers["select"], "t_backward": timers["backward"], "t_forward": timers["forward"],
                "n_iterations": len(res["J_hist"]), "consistency_max_abs": cc["max_abs_diff"], "consistency_rmse": cc["rmse"],
            })

        print(f"\n  {'='*58}\n  State Comparison Summary:\n  {'='*58}")
        print(f"  {'Method':<15} {'T*':<6} {'Final Error':<15} {'Goal State':<20}")
        print(f"  {'-'*58}")
        for method in methods:
            T = results[method]["T_star"]
            err = np.linalg.norm(wrap_error(results[method]["X"][T] - xg, wrap_idx))
            print(f"  {method.capitalize():<15} {T:<6} {err:<15.6f} {str(xg.tolist()):<20}")

        # J_t curves + timing bars
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 2]})
        tgrid = np.arange(1, T_max + 1)
        # Display names for methods
        display_names = {
            "propagator": "Our Method",
            "bruteforce": "Baseline1",
            "onepass":    "Baseline2",
        }
        styles = {
            "propagator": dict(color="#2E86AB", linestyle="-",  linewidth=2.0, marker="o", markersize=3, alpha=0.9),
            "onepass":    dict(color="#A23B72", linestyle="--", linewidth=2.0, marker="s", markersize=3, alpha=0.9),
            "bruteforce": dict(color="#F18F01", linestyle="-.", linewidth=2.0, marker="^", markersize=3, alpha=0.8),
        }
        for m in methods:
            st = styles[m]; markevery = max(1, len(tgrid) // 15)
            ax1.plot(tgrid, results[m]["J_curve"], label=display_names[m], markevery=markevery, **st)
        ax1.set_xlabel("Horizon t (steps)")
        ax1.set_ylabel("Cost $J_t$")
        ax1.set_title(f"{case_name}: Time-Optimal Cost vs. Horizon")
        ax1.legend(fontsize=11, loc="best", framealpha=0.9); ax1.grid(True, alpha=0.3, linestyle="--")

        cats = ["Linearize", "Select", "Backward", "Forward"]
        x = np.arange(len(methods)); width = 0.35  # Narrower bars
        data = {c: [] for c in cats}
        for m in methods:
            tm = results[m]["timers"]
            data["Linearize"].append(tm["linearize"])
            data["Select"].append(tm["select"])
            data["Backward"].append(tm["backward"])
            data["Forward"].append(tm["forward"])
        colors = {"Linearize": "#4ECDC4", "Select": "#FF6B6B", "Backward": "#95E1D3", "Forward": "#FFE66D"}
        bottom = np.zeros(len(methods))
        for c in cats:
            ax2.bar(x, data[c], width, label=c, bottom=bottom, color=colors[c], alpha=0.85, edgecolor="white", linewidth=1.5)
            bottom += data[c]
        for i, m in enumerate(methods):
            total_time = sum(results[m]["timers"].values())
            ax2.text(i, total_time + 0.01, f"{total_time:.3f}s", ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax2.set_xlabel("Method"); ax2.set_ylabel("Time (seconds)")
        ax2.set_title("Computation Time Breakdown")
        ax2.set_xticks(x); ax2.set_xticklabels([display_names[m] for m in methods], fontsize=11)
        ax2.legend(fontsize=10, loc="upper left", framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle="--", axis="y")
        plt.tight_layout()
        fig_path = f"{output_dir}/{case_name}_Jt.png"
        plt.savefig(fig_path, dpi=180, bbox_inches="tight"); plt.close()
        print(f"\n  Saved plot: {fig_path}")

        import pandas as pd
        csv_data = {"t": tgrid}
        for m in methods: csv_data[f"J_{m}"] = results[m]["J_curve"]
        df_curve = pd.DataFrame(csv_data)
        csv_path = f"{output_dir}/{case_name}_Jt.csv"
        df_curve.to_csv(csv_path, index=False)
        print(f"  Saved CSV: {csv_path}")

        traj_path = plot_trajectories(results, methods, case_name, output_dir, state_names, control_names)
        print(f"  Saved trajectories: {traj_path}")

    import pandas as pd
    df_summary = pd.DataFrame(summary_rows)
    summary_path = f"{output_dir}/summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\n{'='*60}\nSummary saved: {summary_path}\n{'='*60}\n")
    return df_summary

if __name__ == "__main__":
    main()
