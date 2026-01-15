# -*- coding: utf-8 -*-
"""Benchmark runner (per-case summaries + tqdm progress).

This script runs a fixed number of randomized trials per case, for three solvers:
  - ourmethod   (information-form propagator sweep)
  - baseline1   (brute-force quadratic-model sweep)
  - baseline2   (paper-style one-pass selection)

Outputs:
  <outdir>/
    summary_all.csv          # all cases concatenated
    summary_agg.csv          # aggregated per (case, solver)
    <CaseName>/summary_all.csv
    <CaseName>/summary_agg.csv

Usage (from this folder):
  python run_suite.py
  python run_suite.py --trials 10 --max-iter 10 --outdir ilqr_results_test
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import wrap_error
from solver import (
    ilqr_timeopt_ourmethod,
    ilqr_timeopt_baseline1,
    ilqr_timeopt_baseline2,
)
from systems import (
    make_double_integrator,
    make_cartpole_swingup,
    make_quadrotor,
    make_segway_balance,
    # make_pointmass_navigation,  # optional (nonconvex obstacles)
)


# -----------------------------------------------------------------------------
# Trial sampling
# -----------------------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def sample_x(base: np.ndarray, sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    base = np.asarray(base, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    if sigma.size == 1:
        sigma = np.full_like(base, float(sigma))
    return base + sigma * rng.standard_normal(base.shape)


# -----------------------------------------------------------------------------
# Case registry
# -----------------------------------------------------------------------------

CaseMaker = Callable[[], Tuple]

CASES: List[Tuple[str, CaseMaker, Dict[str, np.ndarray]]] = [
    ("DoubleIntegrator", make_double_integrator, dict(sigma_x0=np.array([0.2, 0.2]), sigma_xg=np.array([0.0, 0.0]))),
    ("Cartpole_SwingUp", make_cartpole_swingup, dict(sigma_x0=np.array([0.0, 0.0, 0.0, 0.0]), sigma_xg=np.array([0.0, 0.0, 0.0, 0.0]))),
    ("Quadrotor", make_quadrotor, dict(sigma_x0=np.array([0.4, 0.4, 0.4] + [0.0]*9), sigma_xg=np.zeros(12))),
    ("Segway_Balance", make_segway_balance, dict(sigma_x0=np.array([0.02, 0.02, 0.02, 0.02]), sigma_xg=np.array([0.0, 0.0, 0.0, 0.0]))),
]


SOLVERS = {
    "ourmethod": ilqr_timeopt_ourmethod,
    "baseline1": ilqr_timeopt_baseline1,
    "baseline2": ilqr_timeopt_baseline2,
}


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------

def run_case(
    case_name: str,
    maker: CaseMaker,
    sigmas: Dict[str, np.ndarray],
    *,
    outdir: str,
    trials: int,
    seed: int,
    solvers: List[str],
    max_iter: int,
    S_window: int,
    use_central_diff: bool,
    success_tol: float,
) -> pd.DataFrame:
    F, x0_base, xg_base, u_ref, Q, R, alpha, w, N, T_min, T_max, wrap_idx, extra = maker()
    extra_stage_cost = extra.get("extra_stage_cost") if isinstance(extra, dict) else None

    case_dir = os.path.join(outdir, case_name)
    os.makedirs(case_dir, exist_ok=True)

    rng = _rng(seed + hash(case_name) % 10_000)

    rows = []
    n_trials = int(trials)

    p = tqdm(range(n_trials), desc=f"[{case_name}] trials", leave=False)
    for trial in p:
        if trial == 0:
            x0 = np.asarray(x0_base, dtype=float).reshape(-1)
            xg = np.asarray(xg_base, dtype=float).reshape(-1)
        else:
            x0 = sample_x(x0_base, sigmas["sigma_x0"], rng)
            xg = sample_x(xg_base, sigmas["sigma_xg"], rng)

        # per-solver
        for solver_name in solvers:
            fn = SOLVERS[solver_name]

            t0 = time.perf_counter()
            solver_error = None
            try:
                res = fn(
                    F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max,
                    max_iter=max_iter,
                    S_window=S_window,
                    use_central_diff=use_central_diff,
                    wrap_idx=wrap_idx,
                    extra_stage_cost=extra_stage_cost,
                )
            except Exception as e:
                res = None
                solver_error = repr(e)
            t1 = time.perf_counter()

            if res is None:
                rows.append({
                    "case": case_name,
                    "trial": int(trial),
                    "solver": solver_name,
                    "status": "crash",
                    "T_star": int(T_min),
                    "J_star": float("nan"),
                    "total_time": float(t1 - t0),
                    "final_err": float("nan"),
                    "success": False,
                    "n_iter": 0,
                    "solver_error": solver_error,
                })
                p.set_postfix(solver=solver_name, ok=0, T=T_min, J="nan")
                continue

            # solver returned normally
            T_star = int(res.get("T_star", T_min))
            J_star = float(res["J_hist"][-1]) if res.get("J_hist") else float("inf")
            total_time = float(sum(res.get("timers", {}).values())) if isinstance(res.get("timers"), dict) else float(t1 - t0)

            # optional debug signal from baseline2
            if solver_name == "baseline2" and res.get("onepass_error"):
                solver_error = str(res.get("onepass_error"))

            try:
                eT = wrap_error(res["X"][T_star] - xg, wrap_idx)
                final_err = float(np.linalg.norm(np.asarray(eT, dtype=float).reshape(-1)))
            except Exception as e:
                final_err = float("inf")
                solver_error = solver_error or ("final_err failed: " + repr(e))

            success = bool(np.isfinite(J_star) and np.isfinite(final_err) and (final_err <= float(success_tol)))

            rows.append({
                "case": case_name,
                "trial": int(trial),
                "solver": solver_name,
                "status": "ok" if success else "fail",
                "T_star": int(T_star),
                "J_star": float(J_star),
                "total_time": float(total_time),
                "final_err": float(final_err),
                "success": bool(success),
                "n_iter": int(len(res.get("J_hist", []))),
                "solver_error": solver_error,
            })

            p.set_postfix(solver=solver_name, ok=int(success), T=T_star, J=f"{J_star:.3g}")

    df = pd.DataFrame(rows)

    # enrich: best_J per (case,trial)
    best = df.groupby(["case", "trial"])["J_star"].transform("min")
    df["best_J"] = best
    df["cost_ratio_best"] = df["J_star"] / df["best_J"]

    # runtime ratios relative to baseline1 (if present)
    if "baseline1" in solvers:
        base_time = df[df["solver"] == "baseline1"][["case", "trial", "total_time"]].rename(columns={"total_time": "time_base"})
        df = df.merge(base_time, on=["case", "trial"], how="left")
        df["time_ratio_base"] = df["total_time"] / df["time_base"]
    else:
        df["time_base"] = np.nan
        df["time_ratio_base"] = np.nan

    # per-case CSVs
    df.to_csv(os.path.join(case_dir, "summary_all.csv"), index=False)

    agg = (
        df.groupby(["case", "solver"])
          .agg(
              n=("trial", "count"),
              success_rate=("success", "mean"),
              T_median=("T_star", "median"),
              J_median=("J_star", "median"),
              time_median=("total_time", "median"),
              ratio_cost_median=("cost_ratio_best", "median"),
              ratio_time_median=("time_ratio_base", "median"),
          )
          .reset_index()
    )
    agg.to_csv(os.path.join(case_dir, "summary_agg.csv"), index=False)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="ilqr_results", help="output directory")
    ap.add_argument("--trials", type=int, default=25, help="trials per case (same for all cases)")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--max-iter", type=int, default=12, help="max iLQR iterations per run")
    ap.add_argument("--S-window", type=int, default=20, help="onepass search half-window (and initial window for others)")
    ap.add_argument("--use-central-diff", action="store_true", help="use central differences for linearization (slower, but more accurate)")
    ap.add_argument("--success-tol", type=float, default=0.5, help="terminal error norm threshold for success")
    ap.add_argument("--solvers", type=str, default="ourmethod,baseline1,baseline2", help="comma-separated subset")
    ap.add_argument("--cases", type=str, default="", help="comma-separated case names (default: all)")

    args = ap.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    for s in solvers:
        if s not in SOLVERS:
            raise ValueError(f"Unknown solver: {s}. Options: {list(SOLVERS)}")

    if args.cases.strip():
        wanted = set([c.strip() for c in args.cases.split(",") if c.strip()])
        cases = [c for c in CASES if c[0] in wanted]
        if not cases:
            raise ValueError(f"No matching cases in {wanted}. Available: {[c[0] for c in CASES]}")
    else:
        cases = CASES

    all_rows = []
    outer = tqdm(cases, desc="Cases")
    for case_name, maker, sigmas in outer:
        df_case = run_case(
            case_name, maker, sigmas,
            outdir=outdir,
            trials=args.trials,
            seed=args.seed,
            solvers=solvers,
            max_iter=args.max_iter,
            S_window=args.S_window,
            use_central_diff=bool(args.use_central_diff),
            success_tol=args.success_tol,
        )
        all_rows.append(df_case)

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(os.path.join(outdir, "summary_all.csv"), index=False)

    agg_all = (
        df_all.groupby(["case", "solver"])
              .agg(
                  n=("trial", "count"),
                  success_rate=("success", "mean"),
                  T_median=("T_star", "median"),
                  J_median=("J_star", "median"),
                  time_median=("total_time", "median"),
                  ratio_cost_median=("cost_ratio_best", "median"),
                  ratio_time_median=("time_ratio_base", "median"),
              )
              .reset_index()
    )
    agg_all.to_csv(os.path.join(outdir, "summary_agg.csv"), index=False)

    print("\nSaved:")
    print(" ", os.path.join(outdir, "summary_all.csv"))
    print(" ", os.path.join(outdir, "summary_agg.csv"))
    for case_name, _, _ in cases:
        print(" ", os.path.join(outdir, case_name, "summary_all.csv"))


if __name__ == "__main__":
    main()
