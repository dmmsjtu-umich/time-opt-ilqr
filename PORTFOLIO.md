# Propagator-based Time-Optimal iLQR

## Overview

Time-optimal trajectory planning—generating motions that complete a task in the minimum possible time—is a fundamental requirement for agile robotic systems. From autonomous drone racing to emergency collision avoidance in self-driving cars, the ability to jointly optimize the control sequence and the total maneuver duration $T$ is critical for pushing physical limits.

While Differential Dynamic Programming (DDP) and its variant, the iterative Linear Quadratic Regulator (iLQR), have become standard tools for high-dimensional trajectory optimization, they typically assume a fixed planning horizon. Extending these methods to time-optimal control introduces a discrete-continuous optimization challenge: the solver must determine the optimal integer horizon $T^*$ alongside the continuous control inputs.

## The Challenge

**A fundamental bottleneck lies in the structure of the standard Riccati recursion.** In the LQR backward pass, the Value Function $V_k$ is computed recursively starting from a terminal cost anchored at the final time step $T$. Consequently, changing the horizon from $T$ to $T+1$ shifts the boundary condition, invalidating the entire sequence of previously computed Cost-to-Go matrices. This structural dependency prevents the reuse of historical computations across different horizons, forcing the solver to restart the backward pass from scratch for each candidate $T$, resulting in a prohibitive $\mathcal{O}(N^2)$ complexity.

## Our Approach: Time-Varying Propagator

To address the loss of reusability in time-varying systems, we shift from *reusing values* to *reusing mappings*.

![Propagator Flow](paper/figures/propagator_flow.png)

**Figure: Time-varying propagator.** When $g_k$ varies with $k$, we switch to inverse form where each stage is an LFT $\tilde{g}_k$. The composed map $\tilde{g}_{0:k}$ remains an LFT with prefix parameters $(E_{0:k}, F_{0:k}, G_{0:k})$. This enables cheap horizon queries by reusing the composed mapping $\tilde{g}_{0:k}$ instead of reusing $\tilde{P}_k$ values.

Our key idea is to rewrite the map $g_k$ as a new linear fractional transformation (LFT) form $\tilde{g}_{0:k}$, and some of the matrices that help compute $\tilde{g}_{0:k}$ can be reused. As a result, these matrices only need to be computed once for all possible horizons $k=1,2,\cdots,N$, as opposed to be repetitively computed for each possible horizon, which thus saves computational effort.

To apply this logic to iLQR, we introduce an **Augmented State Formulation** that absorbs the time-varying affine linearization terms into a homogeneous coordinate system. This unifies the treatment of linear and nonlinear problems, allowing the propagator to compute the *exact* LQR cost for all horizons in a single $\mathcal{O}(N)$ pass.

## Main Contributions

1. **Propagator-based Horizon Selection:** We develop an LFT-based solver that enables the reuse of backward pass computations, reducing the complexity of horizon selection from $\mathcal{O}(N^2n^3)$ to $\mathcal{O}(Nn^3)$.

2. **Augmented State Formulation:** We propose a state augmentation technique that embeds affine linearization terms into a homogeneous coordinate system, extending the efficient propagator method to general nonlinear iLQR problems.

3. **Performance and Robustness:** We validate our algorithm on four benchmark systems, including a 12-DOF Quadrotor. Experimental results show that our method achieves speedups of up to **43x** compared to brute-force search while guaranteeing global optimality with respect to the linearized model.

## Results

![Quadrotor Hover Results](paper/figures/Quadrotor_Hover_Jt.png)

**Figure: Case Study on Quadrotor Hover.** (Top) Comparison of cost landscapes ($J_t$) computed by different methods. (Bottom) Breakdown of total runtime into linearization, selection, backward, and forward phases.

The top panel shows that our Propagator curve (Blue) overlaps perfectly with the Bruteforce ground truth (Yellow), while the OnePass baseline (Purple) suffers from severe distortions. The bottom panel reveals that our method eliminates the $\mathcal{O}(N^2)$ "Select" phase bottleneck, achieving over **3x speedup** while maintaining exact optimality.
