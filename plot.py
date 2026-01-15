#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def _finite(series):
    s = pd.to_numeric(series, errors="coerce")
    return s[np.isfinite(s)]

def _solver_display_name(s):
    # paper-friendly display names
    if s == "ourmethod":
        return "Ours"
    if s == "baseline1":
        return "Baseline-1 (BF)"
    if s == "baseline2":
        return "Baseline-2 (OnePass)"
    return str(s)

def _case_display_name(c):
    return str(c).replace("_", " ")

def _order_from_list(all_items, preferred):
    out = [x for x in preferred if x in all_items]
    out += [x for x in all_items if x not in out]
    return out

def _detect_key(df):
    """
    Support both summary formats:
      - (case, trial)
      - (case, start_id, goal_id)
    """
    if {"case", "trial"}.issubset(df.columns):
        return ["case", "trial"]
    if {"case", "start_id", "goal_id"}.issubset(df.columns):
        return ["case", "start_id", "goal_id"]
    if "case" in df.columns:
        return ["case"]
    raise ValueError("CSV must contain at least a 'case' column.")

def _compute_ratios(df):
    """
    Robustly ensure these columns exist:
      - best_J (per scenario = per key)
      - cost_ratio_best = J_star / best_J
      - time_base (baseline1 runtime per scenario)
      - time_ratio_base = total_time / time_base
    """
    df = df.copy()
    key = _detect_key(df)

    # best achieved cost per scenario (min across solvers)
    df["best_J"] = df.groupby(key)["J_star"].transform("min")
    df["cost_ratio_best"] = df["J_star"] / df["best_J"]

    # baseline1 runtime per scenario (NO merge; map back by index)
    df["time_base"] = np.nan
    df["time_ratio_base"] = np.nan

    solvers = set(df["solver"].astype(str).unique())
    if "baseline1" in solvers:
        base_series = (
            df[df["solver"] == "baseline1"]
            .groupby(key)["total_time"]
            .first()
        )
        # map back
        idx = df.set_index(key).index
        df["time_base"] = idx.map(base_series)
        df["time_ratio_base"] = df["total_time"] / df["time_base"]

    return df

def _filter_success_only(df):
    if "success" not in df.columns:
        return df
    return df[df["success"] == True].copy()


# -----------------------------
# Plot primitives
# -----------------------------
def boxplot_groups(ax, data_by_group, tick_labels, ylabel=None, ylog=False, title=None):
    ax.boxplot(
        data_by_group,
        tick_labels=tick_labels,   # Matplotlib 3.9+
        showfliers=False,
        widths=0.6,
    )
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    if ylog:
        ax.set_yscale("log")

def savefig(fig, path, dpi=220):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)

def _paper_main(df_succ, cases_order, solvers_order, outdir):
    """
    One compact "paper main" figure with 2 panels:
      (a) Runtime ratio vs Baseline-1
      (b) Cost ratio vs best-achieved
    Each point = median, whisker = IQR [25%, 75%].
    """
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
    })

    cases_disp = [_case_display_name(c) for c in cases_order]

    def stats(metric):
        rows = []
        for c in cases_order:
            for s in solvers_order:
                sub = df_succ[(df_succ["case"] == c) & (df_succ["solver"] == s)]
                vals = _finite(sub[metric])
                if len(vals) == 0:
                    rows.append((c, s, np.nan, np.nan, np.nan))
                    continue
                q1 = np.percentile(vals, 25)
                med = np.percentile(vals, 50)
                q3 = np.percentile(vals, 75)
                rows.append((c, s, med, q1, q3))
        return pd.DataFrame(rows, columns=["case","solver","med","q1","q3"])

    st_rt = stats("time_ratio_base")
    st_cs = stats("cost_ratio_best")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    x = np.arange(len(cases_order))
    offsets = np.linspace(-0.22, 0.22, len(solvers_order))

    # (a) runtime ratio (log)
    ax = axes[0]
    for i, s in enumerate(solvers_order):
        sub = st_rt[st_rt["solver"] == s].set_index("case")
        y  = np.array([sub.loc[c, "med"] if c in sub.index else np.nan for c in cases_order], float)
        q1 = np.array([sub.loc[c, "q1"] if c in sub.index else np.nan for c in cases_order], float)
        q3 = np.array([sub.loc[c, "q3"] if c in sub.index else np.nan for c in cases_order], float)
        yerr = np.vstack([y - q1, q3 - y])
        ax.errorbar(x + offsets[i], y, yerr=yerr, fmt="o", capsize=4, label=_solver_display_name(s))

    ax.axhline(1.0, linewidth=1.0, alpha=0.4)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(cases_disp)
    ax.set_ylabel("Runtime / Baseline-1 (BF)")
    ax.set_title("(a) Runtime")
    ax.grid(True, axis="y", alpha=0.25)

    # (b) cost ratio (plain formatting)
    ax = axes[1]
    for i, s in enumerate(solvers_order):
        sub = st_cs[st_cs["solver"] == s].set_index("case")
        y  = np.array([sub.loc[c, "med"] if c in sub.index else np.nan for c in cases_order], float)
        q1 = np.array([sub.loc[c, "q1"] if c in sub.index else np.nan for c in cases_order], float)
        q3 = np.array([sub.loc[c, "q3"] if c in sub.index else np.nan for c in cases_order], float)
        yerr = np.vstack([y - q1, q3 - y])
        ax.errorbar(x + offsets[i], y, yerr=yerr, fmt="o", capsize=4, label=_solver_display_name(s))

    ax.axhline(1.0, linewidth=1.0, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(cases_disp)
    ax.set_ylabel("J / best")
    ax.set_title("(b) Cost")
    ax.grid(True, axis="y", alpha=0.25)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))

    out = os.path.join(outdir, "paper_main.png")
    savefig(fig, out)
    return out

def _boxplot_by_case(df_succ, metric, ylabel, title, outpath, cases_order, solvers_order, ylog=False):
    n = len(cases_order)
    fig, axes = plt.subplots(1, n, figsize=(4.0*n, 3.4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, c in zip(axes, cases_order):
        sub = df_succ[df_succ["case"] == c]
        data, ticks = [], []
        for s in solvers_order:
            vals = _finite(sub[sub["solver"] == s][metric])
            data.append(vals.values)
            ticks.append(_solver_display_name(s))
        boxplot_groups(ax, data, ticks, ylabel=None, ylog=ylog, title=_case_display_name(c))
        ax.tick_params(axis="x", rotation=0)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, y=1.03)
    savefig(fig, outpath)

def _boxplot_cost_ratio(df_succ, outdir, cases_order, solvers_order):
    outpath = os.path.join(outdir, "box_cost_ratio_best.png")
    _boxplot_by_case(
        df_succ,
        metric="cost_ratio_best",
        ylabel="J / best",
        title="Cost ratio vs best-achieved (success only)",
        outpath=outpath,
        cases_order=cases_order,
        solvers_order=solvers_order,
        ylog=False,
    )

def _boxplot_runtime_ratio(df_succ, outdir, cases_order, solvers_order):
    # if runtime ratio is all NaN, skip
    if "time_ratio_base" not in df_succ.columns:
        return
    if _finite(df_succ["time_ratio_base"]).shape[0] == 0:
        print("Skip runtime-ratio plots: time_ratio_base is all NaN (baseline1 missing or time_base missing).")
        return

    outpath = os.path.join(outdir, "box_runtime_ratio.png")
    _boxplot_by_case(
        df_succ,
        metric="time_ratio_base",
        ylabel="time / baseline1",
        title="Runtime ratio vs baseline1 (success only)",
        outpath=outpath,
        cases_order=cases_order,
        solvers_order=solvers_order,
        ylog=True,
    )

def _boxplot_T_star(df_succ, outdir, cases_order, solvers_order):
    if "T_star" not in df_succ.columns:
        return
    outpath = os.path.join(outdir, "box_T_star.png")
    _boxplot_by_case(
        df_succ,
        metric="T_star",
        ylabel="T*",
        title="Selected horizon T* distribution (success only)",
        outpath=outpath,
        cases_order=cases_order,
        solvers_order=solvers_order,
        ylog=False,
    )


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to summary_all.csv. If omitted, uses <outdir>/summary_all.csv")
    parser.add_argument("--outdir", type=str, default=".",
                        help="Output directory. Plots saved to <outdir>/plots/")
    parser.add_argument("--cases", type=str, default="",
                        help="Comma-separated cases. Default: all.")
    parser.add_argument("--solvers", type=str, default="ourmethod,baseline1,baseline2",
                        help="Comma-separated solver order.")
    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir)
    plots_dir = _ensure_dir(os.path.join(outdir, "plots"))

    csv_path = args.csv if args.csv is not None else os.path.join(outdir, "summary_all.csv")
    csv_path = os.path.abspath(csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # legacy rename
    if "method" in df.columns and "solver" not in df.columns:
        df = df.rename(columns={"method": "solver"})

    need = {"case", "solver", "J_star", "total_time"}
    missing = sorted(list(need - set(df.columns)))
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # optional case filter
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    if cases:
        df = df[df["case"].isin(cases)].copy()

    # solver order
    solvers_present = sorted(df["solver"].astype(str).unique().tolist())
    solvers_order = [s.strip() for s in args.solvers.split(",") if s.strip()]
    solvers_order = _order_from_list(solvers_present, solvers_order)

    cases_order = sorted(df["case"].unique().tolist())

    # compute ratios robustly
    df = _compute_ratios(df)

    # success only
    df_succ = _filter_success_only(df)

    # plots
    _paper_main(df_succ, cases_order, solvers_order, plots_dir)
    _boxplot_cost_ratio(df_succ, plots_dir, cases_order, solvers_order)
    _boxplot_runtime_ratio(df_succ, plots_dir, cases_order, solvers_order)
    _boxplot_T_star(df_succ, plots_dir, cases_order, solvers_order)

    print("\nDone. Plots saved in:", plots_dir)


if __name__ == "__main__":
    main()