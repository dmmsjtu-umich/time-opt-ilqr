# HOP: Fast Differential Dynamic Programming for Horizon-Optimal Trajectory Planning

## Overview

This paper considers a **Horizon-Optimal Control** problem that seeks a dynamically feasible trajectory while minimizing the planning horizon, which is a fundamental problem in robotics with numerous applications. While many famous optimal control methods, such as LQR, iLQR/DDP, are well studied and deployed on various robots, they often have a fixed planning horizon, and their horizon-optimal counterparts are still undiscovered.

The best result in the literature solves the horizon-optimal LQR problem by shifting the horizon and reusing the value functions computed by the Riccati recursion. However, this approach is limited to LQR with **time-invariant dynamics and costs only**.

## The Challenge

**A fundamental bottleneck lies in the structure of the standard Riccati recursion.** In the LQR backward pass, the Value Function $V_k$ is computed recursively starting from a terminal cost anchored at the final time step $T$. For non-stationary dynamics or costs, the value functions are time-varying and thus cannot be reused, and the idea of "shifting the horizon" fails. A naive approach requires solving the Riccati equation repeatedly for every horizon, which has a runtime complexity of $\mathcal{O}(N^2n^3)$.

## Our Approach: Linear Fractional Transformation (LFT)

![Header Figure](paper/figures/header.png)

**Figure: Quadrotor Results.** For a quadrotor dynamics with 12 degrees of freedom, our HOP-DDP finds the best solution trajectory with horizon $T^* = 26$ in 2.9 seconds while the DDP based on time-invariant LQR (OP) and NLP converges to local minima $T^* = 48, 30$ in 3.8 and 71.8 seconds respectively.

The key insight in HOP is that the Riccati recursion can be reformulated into a form of **Linear Fractional Transformation (LFT)**, which enjoys the structure that enables efficient computational reuse even for non-stationary dynamics and costs.

![Algorithm Flow](paper/figures/alg_flow.png)

**Figure: HOP-LQR Algorithm.** The system matrices are used to compute matrices $(E_k, F_k, G_k)$ and $(\bar{E}_k, \bar{F}_k, \bar{G}_k)$. The composed maps $\tilde{g}_{0:k}(\cdot)$ enable direct evaluation of the inverse of initial cost-to-go for any horizon $k$.

Based on this idea, we develop **HOP-LQR** that can solve Horizon-Optimal Time-Varying LQR problem to optimality, and we show that its runtime complexity is same as a regular Riccati recursion for the basic LQR problem. Based on HOP-LQR, we further develop **HOP-DDP** by introducing an augmented state space formulation, which allows solving horizon-optimal OC problems with general nonlinear dynamics and non-quadratic costs efficiently.

## Main Contributions

1. **HOP-LQR Algorithm:** We develop an LFT-based solver that enables the reuse of backward pass computations, reducing the complexity of horizon selection from $\mathcal{O}(N^2n^3)$ to $\mathcal{O}(Nn^3)$.

2. **HOP-DDP Algorithm:** We propose an augmented state space formulation that embeds affine linearization terms into a homogeneous coordinate system, extending the efficient method to general nonlinear dynamics and non-quadratic costs.

3. **Performance:** Our approach always finds the same optimal solution as a naive brute force baseline method, while running up to **40× faster**. For nonlinear dynamics, our method always finds better solutions than approximation using time-invariant LQR.

## Results

![Experimental Results](paper/figures/exp1_all.png)

**Figure: Overall Performance.** (a) Runtime comparison in log scale. (b) Speedup relative to Brute Force baseline. (c) Success rates of different algorithms. (d) Solution cost increases relative to Brute Force baseline.

### Case Study: Quadrotor Hovering

![Cost Curve](paper/figures/Quadrotor_Hover_cost_curve.png)

**Figure: Cost values for various horizons.** Our method and Baseline-1 (BF) both find optimal horizon $T^* = 32$ with costs $J_{32}^{ours} \approx 484.79$, $J_{32}^{BF} \approx 484.80$, while Baseline-2 (OP) converges to a local minimum with longer horizon ($T \approx 74$) and 5.97% higher cost.

![Runtime Breakdown](paper/figures/Quadrotor_Hover_timing.png)

**Figure: Runtime breakdown.** Ours and Baseline-2 (OP) run faster than Baseline-1 (BF) as they bypass the expensive computation for horizon selection.
