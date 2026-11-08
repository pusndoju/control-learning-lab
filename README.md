# Control Learning Lab

A personal self-study lab for **control and automation**. I read textbooks and papers chapter by chapter, then re-implement the key algorithms as small, self-contained simulations, in Python or MATLAB/Simulink, to see them run and to check the conditions that the theory relies on.

## About me

- Bachelor's degree from **Eastern International University (EIU)**, Vietnam.
- Preparing for a **Master's in Control and Automation**.
- **Target thesis direction:** deep reinforcement learning (DRL) combined with a **predictive safety filter (PSF)**, that is, learning-based control that stays safe under constraints.

## Learning roadmap

The topics are ordered backwards from the thesis direction:

| Stage | Topics | Status |
|---|---|---|
| 1. RL foundations | MDPs, Bellman equations, dynamic programming, Monte Carlo, stochastic approximation, TD learning | In progress: chapters 1–7 done |
| 2. Deep RL | Function approximation, DQN, policy gradient, actor-critic, SAC / PPO | Next |
| 3. Optimal and predictive control | LQR / Riccati, Lyapunov stability, constrained MPC (terminal sets, recursive feasibility) | Planned |
| 4. Safe learning | Control barrier functions (CBF), predictive safety filters (PSF) | Planned |
| 5. RL + PSF | Combining deep RL with a safety filter on a robot | Thesis |

Side topics such as fuzzy control, state estimation (EKF), and embedded code generation from Simulink are added when they come up.

## Repository layout

```
control-learning-lab/
├─ README.md                  this page
├─ linux/                     Python simulations (Ubuntu / Linux VM)
│  ├─ README.md               environment setup and how to run
│  ├─ .vscode/                VS Code interpreter + F5 launch config
│  └─ <book-or-topic>/        one folder per book or topic, each with its own README
└─ matlab/                    MATLAB / Simulink simulations (coming soon)
```

Each level has its own README. For Python, start with [`linux/README.md`](linux/README.md) to set up the environment, then open a book folder.

## Contents

| Source | Platform | Folder | Covered |
|---|---|---|---|
| *Mathematical Foundations of Reinforcement Learning*, Shiyu Zhao | Python (Gymnasium) | [`linux/mathematical-foundations-rl/`](linux/mathematical-foundations-rl/) | Ch. 1–7: Bellman equation, value/policy iteration, Monte Carlo, Robbins-Monro and SGD, TD learning (Sarsa, n-step Sarsa, Q-learning) |
| Fuzzy PID control of an inverted pendulum, with C code generation | MATLAB / Simulink | coming soon | |
| 6-DOF robot arm with EKF and PID control | MATLAB / Simulink | coming soon | |

## Tools

| | Used for |
|---|---|
| **Python** with NumPy, Matplotlib, [Gymnasium](https://gymnasium.farama.org/) | RL and anything that needs many episodes and seeds |
| **MATLAB / Simulink** (Control, MPC, RL, Fuzzy Logic toolboxes, Embedded Coder) | Control design, hybrid or multi-rate systems, code generation for embedded targets |
| **VMware Ubuntu VM + VS Code** | Running the Python simulations (setup in [`linux/README.md`](linux/README.md)) |

## How each simulation is written

- **One algorithm = one file**, named `chXX_<algorithm>_<environment>`, so each file maps to one section of its source.
- The code follows the source's notation. Lines that implement the algorithm cite it in a comment, for example `[Book eq. 7-18]` or `[Book Alg. 4.1]`.
- Each book folder's README lists, for every file, the question the simulation answers, reference results, and where my results differ from the book and why.
- Every file runs as-is with its default settings. Outputs (plots, saved tables) go to a `results/` folder next to the file and are not tracked by Git.

## Acknowledgments

Inspired by the textbooks listed above and by lecture and tutorial videos on YouTube. This repository contains only my own code and notes, not any book content. Please read the original sources.
