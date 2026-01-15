# -*- coding: utf-8 -*-
"""Benchmark systems (discrete-time) for time-optimal iLQR experiments.

These are small, self-contained dynamics/cost definitions inspired by:
"Optimal-Horizon Model-Predictive Control with Differential Dynamic Programming".

Notes:
- All dynamics are implemented in discrete time (Euler integration) and expose
  `F.dt` for convenience.
- Some systems (notably the quadrotor with Euler angles) can become numerically
  unstable during line-search / finite-difference linearization. We add small,
  *physically reasonable* clamps to keep intermediate rollouts bounded. These
  clamps are meant as numerical safeguards; they do not change the core solver
  logic.
"""

from __future__ import annotations

import math
import numpy as np
from utils import angle_normalize


# =============================================================================
# 1) Linear system: 1D double integrator
# =============================================================================

def make_double_integrator(dt: float = 0.05, N: int = 120):
    """x=[pos, vel], u=[acc]."""
    def F(x, u):
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float).reshape(-1)
        return np.array([x[0] + dt * x[1], x[1] + dt * u[0]], dtype=float)

    F.dt = dt

    x0 = np.array([1.0, 0.0], dtype=float)
    xg = np.array([2.0, 0.0], dtype=float)
    u_ref = np.array([0.0], dtype=float)

    Q = np.diag([1.0, 0.1]).astype(float)
    R = np.array([[1e-2]], dtype=float)

    alpha = 50.0
    w = 0.02

    T_min, T_max = 10, 80
    wrap_idx = []
    extra = None
    return F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max, wrap_idx, extra


# =============================================================================
# 2) Cart-pole swing-up
# =============================================================================

def make_cartpole_swingup(dt: float = 0.02, N: int = 360):
    """Cart-pole swing-up.

    State: [cart_pos, cart_vel, theta, theta_dot]
      - theta is stored so theta=0 is *down*, theta=pi is *upright*.
    Control: [force]
    """
    g = 9.81
    m_cart = 1.0
    m_pole = 0.1
    length = 0.5  # half-length

    total_mass = m_cart + m_pole
    polemass_length = m_pole * length

    def F(x, u):
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float).reshape(-1)
        x_pos, x_dot, th, th_dot = x
        force = float(u[0])

        # shift angle so internal dynamics match standard form (theta=0 upright)
        th_u = th - math.pi
        costh = math.cos(th_u)
        sinth = math.sin(th_u)

        temp = (force + polemass_length * th_dot * th_dot * sinth) / total_mass
        denom = length * (4.0 / 3.0 - m_pole * costh * costh / total_mass)

        th_acc = (g * sinth - costh * temp) / denom
        x_acc = temp - polemass_length * th_acc * costh / total_mass

        xn = np.array([
            x_pos + dt * x_dot,
            x_dot + dt * x_acc,
            angle_normalize(th + dt * th_dot),
            th_dot + dt * th_acc,
        ], dtype=float)
        return xn

    F.dt = dt

    x0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    xg = np.array([0.0, 0.0, math.pi, 0.0], dtype=float)
    u_ref = np.array([0.0], dtype=float)

    Q = np.diag([0.01, 0.2, 0.0, 0.2]).astype(float)
    R = np.array([[0.02]], dtype=float)

    alpha = np.diag([5.0, 5.0, 800.0, 40.0]).astype(float)
    w = 0.03

    T_min, T_max = 40, 320
    wrap_idx = [2]  # theta
    extra = None
    return F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max, wrap_idx, extra


# =============================================================================
# 3) Quadrotor (12D Euler-angle model)
# =============================================================================

def make_quadrotor(dt: float = 0.05, N: int = 160):
    """Simple 12D quadrotor with Euler angles.

    State:
      x = [pos(3), vel(3), euler(3), omega(3)]
    Control:
      u = [thrust, tau_x, tau_y, tau_z]

    IMPORTANT:
      - We keep the dynamics *smooth* (no hard saturation inside F), because
        iLQR relies on local derivatives.
      - To prevent the classic Euler-angle singularity near pitch=±pi/2 (where
        sec(pitch) blows up), we add a *guard*: if |cos(pitch)| is too small we
        return NaNs so the line-search / candidate rollout is rejected.

    This preserves the good convergence behavior you had before, while avoiding
    the overflow cascades that can happen when intermediate rollouts hit the
    singular region.
    """
    m, g = 1.0, 9.81
    Ix, Iy, Iz = 0.02, 0.02, 0.04
    kv, kw = 0.05, 0.01

    I = np.diag([Ix, Iy, Iz]).astype(float)
    I_inv = np.diag([1.0 / Ix, 1.0 / Iy, 1.0 / Iz]).astype(float)

    def rotm(phi, th, psi):
        s, c = np.sin, np.cos
        Rz = np.array([[c(psi), -s(psi), 0.0],
                       [s(psi),  c(psi), 0.0],
                       [0.0,     0.0,    1.0]], dtype=float)
        Ry = np.array([[c(th), 0.0, s(th)],
                       [0.0,   1.0, 0.0],
                       [-s(th),0.0, c(th)]], dtype=float)
        Rx = np.array([[1.0, 0.0,      0.0],
                       [0.0, c(phi),  -s(phi)],
                       [0.0, s(phi),   c(phi)]], dtype=float)
        return Rz @ Ry @ Rx

    def Tmat(phi, th):
        s, c, t = np.sin, np.cos, np.tan
        sec = lambda x: 1.0 / np.cos(x)
        return np.array([[1.0, s(phi) * t(th),       c(phi) * t(th)],
                         [0.0, c(phi),             -s(phi)],
                         [0.0, s(phi) * sec(th),    c(phi) * sec(th)]], dtype=float)

    # guards (very loose; only to avoid numerical blow-ups)
    cos_pitch_min = 1e-3      # reject if too close to Euler singularity
    omg_abs_max   = 1e3       # rad/s guard to avoid overflow in cross products
    state_norm_max = 1e6      # generic divergence guard

    def F(x, u):
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        if x.size != 12 or u.size != 4:
            raise ValueError("Quadrotor expects x in R^12 and u in R^4")
        if (not np.all(np.isfinite(x))) or (not np.all(np.isfinite(u))):
            return np.full(12, np.nan, dtype=float)
        if float(np.linalg.norm(x)) > state_norm_max:
            return np.full(12, np.nan, dtype=float)

        pos = x[0:3]
        vel = x[3:6]
        phi, th, psi = x[6:9]
        omg = x[9:12]

        # Euler singularity guard
        if abs(float(np.cos(th))) < cos_pitch_min:
            return np.full(12, np.nan, dtype=float)

        # absurd omega guard (prevents overflow cascades)
        if np.any(np.abs(omg) > omg_abs_max):
            return np.full(12, np.nan, dtype=float)

        thrust = float(u[0])
        tau = np.asarray(u[1:4], dtype=float)

        Rb = rotm(phi, th, psi)
        e3 = np.array([0.0, 0.0, 1.0], dtype=float)

        acc = (Rb @ (e3 * thrust)) / m - np.array([0.0, 0.0, g]) - kv * vel
        eulerdot = Tmat(phi, th) @ omg
        omgdot = I_inv @ (tau - np.cross(omg, I @ omg)) - kw * omg

        xdot = np.zeros(12, dtype=float)
        xdot[0:3] = vel
        xdot[3:6] = acc
        xdot[6:9] = eulerdot
        xdot[9:12] = omgdot

        xn = x + dt * xdot
        return xn

    F.dt = dt

    x0 = np.zeros(12, dtype=float)
    x0[0:3] = [2.0, 2.0, 2.0]  # start away from origin
    xg = np.zeros(12, dtype=float)

    # nominal control (hover thrust)
    u_ref = np.array([m * g, 0.0, 0.0, 0.0], dtype=float)

    Q = np.diag([5, 5, 5, 1, 1, 1, 20, 20, 10, 1, 1, 1]).astype(float)
    R = np.diag([1e-3, 1e-2, 1e-2, 1e-2]).astype(float)

    alpha = 300.0  # scalar terminal weight (reference design)
    w = 0.005

    T_min, T_max = 40, 160
    wrap_idx = [6, 7, 8]  # Euler angles
    extra = None
    return F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max, wrap_idx, extra


# =============================================================================
# 4) 2D point-mass navigation with obstacle costs (optional)
# =============================================================================

def make_pointmass_navigation(dt: float = 0.05, N: int = 240):
    """2D double-integrator navigation with soft obstacle penalties."""
    def F(x, u):
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float).reshape(-1)
        px, py, vx, vy = x
        ax, ay = u
        return np.array([
            px + dt * vx,
            py + dt * vy,
            vx + dt * ax,
            vy + dt * ay,
        ], dtype=float)

    F.dt = dt

    x0 = np.array([-2.0, -2.0, 0.0, 0.0], dtype=float)
    xg = np.array([ 2.0,  2.0, 0.0, 0.0], dtype=float)
    u_ref = np.array([0.0, 0.0], dtype=float)

    Q = np.diag([0.0, 0.0, 0.15, 0.15]).astype(float)
    R = np.diag([0.05, 0.05]).astype(float)
    alpha = np.diag([250.0, 250.0, 30.0, 30.0]).astype(float)
    w = 0.06

    T_min, T_max = 30, 220
    wrap_idx = []

    obstacles = [
        dict(center=np.array([-1.0, -0.5]), radius=0.65, weight=6.0),
        dict(center=np.array([ 0.0,  0.2]), radius=0.70, weight=6.0),
        dict(center=np.array([ 1.0,  1.0]), radius=0.65, weight=6.0),
    ]

    def extra_stage_cost(x, u):
        p = np.asarray(x[:2], dtype=float)
        c = 0.0
        cx = np.zeros(4, dtype=float)
        cxx = np.zeros((4, 4), dtype=float)

        for obs in obstacles:
            o = obs["center"]
            r = float(obs["radius"])
            w_i = float(obs["weight"])
            d = p - o
            s = float(d @ d)
            base = math.exp(-s / (2.0 * r * r))
            ci = w_i * base

            gi = -(ci / (r * r)) * d
            Hi = ci * (np.outer(d, d) / (r ** 4) - np.eye(2) / (r * r))

            c += ci
            cx[:2] += gi
            cxx[:2, :2] += Hi

        return c, cx, cxx

    extra = dict(obstacles=obstacles, extra_stage_cost=extra_stage_cost)
    return F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max, wrap_idx, extra


# =============================================================================
# 5) Segway balance
# =============================================================================

def make_segway_balance(dt: float = 0.02, N: int = 240):
    """Segway (inverted pendulum on a wheel) – local stabilization benchmark."""
    g = 9.81
    r = 0.15
    M = 1.0
    m = 2.0
    l = 0.5
    I = (1.0 / 3.0) * m * l * l
    a1 = M + m
    a2 = m * l
    a3 = I + m * l * l
    Den = a1 * a3 - a2 * a2

    A_tau = a3 / (r * Den) - a2 / Den
    A_th  = -(a2 * m * g * l) / Den
    B_tau = -a2 / (r * Den) + a1 / Den
    B_th  = (a1 * m * g * l) / Den

    def F(x, u):
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float).reshape(-1)
        x_pos, x_dot, th, th_dot = x
        tau = float(u[0])
        xdd  = A_tau * tau + A_th * th
        thdd = B_tau * tau + B_th * th
        return np.array([
            x_pos + dt * x_dot,
            x_dot + dt * xdd,
            angle_normalize(th + dt * th_dot),
            th_dot + dt * thdd,
        ], dtype=float)

    F.dt = dt

    x0 = np.array([0.05, 0.0, 0.08, 0.0], dtype=float)
    xg = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    u_ref = np.array([0.0], dtype=float)

    Q = np.diag([1.0, 0.1, 25.0, 1.0]).astype(float)
    R = np.array([[0.25]], dtype=float)
    alpha = np.diag([20.0, 2.0, 250.0, 10.0]).astype(float)
    w = 1e-4

    T_min, T_max = 40, 200
    wrap_idx = [2]
    extra = None
    return F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max, wrap_idx, extra