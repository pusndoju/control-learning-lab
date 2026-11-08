# Mathematical Foundations of Reinforcement Learning: chapters 1–7

My simulations written while reading *Mathematical Foundations of Reinforcement Learning* by **Shiyu Zhao**. Each key algorithm of the book is one file. Code comments cite the matching equation or algorithm (`[Book eq. x-y]`, `[Book Alg. x.y]`).

Setup and how to run: see the [Linux README](../README.md).

## Environment

- **`FrozenLake-v1` 4×4** from Gymnasium is used in chapters 2, 4, 5 and 7, with γ = 0.9. Set `is_slippery=True` for stochastic transitions (1/3 intended direction, 1/3 to each side).
- Chapter 6 uses plain numpy.
- Model-based methods (chapters 2 and 4) read p(s'|s,a) from `env.unwrapped.P`.
- Model-free methods (chapters 5 and 7) only call `env.step()`. The model is used only to compute v\* as a reference answer.

**Differences from the book's 5×5 grid world:** in FrozenLake the goal and the holes are terminal states, and the reward is +1 at the goal and 0 everywhere else (no −1 penalties). So v ≤ 1 and the numbers differ from the book's figures, while every formula is the same.

## Files

| Ch. | File | Algorithm | Question the simulation answers |
|---|---|---|---|
| 1 | (none) | Basic concepts | (notes only) |
| 2 | `ch02_policy_evaluation_frozenlake.py` | Bellman equation: closed form (I−γP_π)⁻¹r_π and iterative solution (2.11) | Do both give the same v_π? Does the error shrink like γ^k (Box 2.1)? |
| 3 | (merged into value iteration) | BOE, contraction mapping (Thm 3.1–3.5) | |
| 4 | `ch04_value_iteration_frozenlake.py` | Value iteration (Alg 4.1) | Geometric convergence from any v₀? |
| 4 | `ch04_policy_iteration_frozenlake.py` | Policy iteration (Alg 4.2) | Is v_{π_k} monotonically increasing (Thm 4.1)? |
| 4 | `ch04_truncated_policy_iteration_frozenlake.py` | Truncated PI (Alg 4.3) | Outer iterations vs. evaluation sweeps (Fig 4.5) |
| 5 | `ch05_mc_basic_frozenlake.py` | MC Basic (Alg 5.1) | What goes wrong when episodes are too short (Fig 5.4)? |
| 5 | `ch05_mc_exploring_starts_frozenlake.py` | MC Exploring Starts (Alg 5.2) | How much more sample-efficient is every-visit? |
| 5 | `ch05_mc_epsilon_greedy_frozenlake.py` | MC ε-Greedy (Alg 5.3) | The optimal policy in Π_ε differs from the optimal policy in Π |
| 6 | `ch06_robbins_monro_root_finding.py` | Robbins-Monro (6.5) | What happens when each condition of Thm 6.1 is violated? |
| 6 | `ch06_sgd_mean_estimation.py` | BGD / MBGD / SGD (Sec 6.4.4) | How does the mini-batch size affect convergence (Fig 6.5)? |
| 7 | `ch07_td_state_value_frozenlake.py` | TD(0) (7.1) | Constant α vs. α = 1/N(s) (Thm 7.1) |
| 7 | `ch07_sarsa_frozenlake.py` | Sarsa (Alg 7.1) | Sarsa only finds a path from the start state |
| 7 | `ch07_nstep_sarsa_frozenlake.py` | n-step Sarsa (7.17) | n = 1, 3, 10: bias vs. variance |
| 7 | `ch07_q_learning_frozenlake.py` | Q-learning off-policy (Alg 7.3) and on-policy (Alg 7.2) | Learning the optimal policy from random behavior data |

Run them in table order. Each chapter builds on the previous one: chapter 2 gives policy evaluation, chapter 4 adds policy improvement, chapter 5 removes the model, chapter 6 provides the tools (RM/SGD), and chapter 7 combines everything into TD learning.

## Reference results

Non-slippery FrozenLake unless stated otherwise.

- **Ch 2:** v_π(0) = 0.590 for the "good" policy and ≈ 0.004 for the random policy. The iterative solution matches the closed form to ~1e-8.
- **Ch 4:** VI converges in 7 iterations. PI gives Σv = 0 → 1 → 2.8 → … → 8.44. All three algorithms return the same π\*. Slippery: truncated PI with j = 1 / 3 / 10 / 100 needs 92 / 32 / 11 / 5 outer iterations, and π\* reaches the goal about 73% of the time.
- **Ch 5:** MC Basic needs `episode_length` ≥ 6 (the minimum number of steps from state 0 to the goal), otherwise the error gets stuck at 0.9 / 0.73 / 0.59. ε-greedy with ε = 0.5 picks R at state 0, unlike π\*.
- **Ch 6:** for g(w) = w³ − 5, step sizes a_k = 1/k and 1/k² **diverge** because g' is unbounded (condition (a) of Thm 6.1 fails). a_k = 0.2/k converges to 1.71. Note: the book reports convergence in Fig 6.3 with w₁ = 0 and a_k = 1/k, but every seed I tried diverged.
- **Ch 7:** off-policy Q-learning with uniform behavior reaches v\* (RMSE → 0). Behavior ε = 0.1 gets stuck because it explores too little. Unlike Fig 7.5, ε = 0.5 is faster than ε = 1 here: FrozenLake ends at holes, so a random walk rarely reaches the states near the goal.

## Convergence conditions to remember

| Algorithm | Condition |
|---|---|
| Policy evaluation, VI | γ < 1, so the operator is a γ-contraction. Geometric convergence from any initial value |
| PI, truncated PI | Values increase monotonically (Thm 4.1, Prop 4.1) |
| MC | Enough episodes, long enough episodes, every (s,a) visited (exploring starts or ε-greedy) |
| RM, SGD, TD, Sarsa, Q-learning | Σα = ∞ and Σα² < ∞ for **every** state or (s,a), so every (s,a) must be visited infinitely often. A constant α converges only in expectation |
