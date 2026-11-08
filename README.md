# RL Book Simulations

Learning by reading and simulating: I read a textbook chapter by chapter, then re-implement its key algorithms as small, self-contained simulations to see them run and to check the convergence conditions stated in the book.

The project was inspired by the book **Mathematical Foundations of Reinforcement Learning** by **Shiyu Zhao**, together with lecture and tutorial videos on YouTube. This repository contains only my own code and notes, not the book's content. Please read the original book.

## Repository layout

```
rl-book-simulations/
├─ README.md                  this page
├─ linux/                     Python simulations (run on Ubuntu / a Linux VM)
│  ├─ README.md               environment setup and how to run
│  ├─ .vscode/                VS Code interpreter + F5 launch config
│  └─ mathematical-foundations-rl/
│     └─ README.md            chapter-by-chapter guide for this book
└─ matlab/                    MATLAB / Simulink simulations (coming later)
```

Each level has its own README. Start with the platform folder (`linux/`) for setup, then open the book folder.

## Books

| Book | Platform | Folder | Chapters |
|---|---|---|---|
| *Mathematical Foundations of Reinforcement Learning*, Shiyu Zhao | Python | [`linux/mathematical-foundations-rl/`](linux/mathematical-foundations-rl/) | 1–7: Bellman equation, value/policy iteration, Monte Carlo, stochastic approximation, TD learning (Sarsa, Q-learning) |

More books, and MATLAB/Simulink simulations, will be added over time.

## Conventions

- **One algorithm = one file**, named `chXX_<algorithm>_<environment>`, so each file maps to one section of the book.
- Code follows the book's notation. Lines that implement the algorithm cite the book in a comment, for example `[Book eq. 7-18]` or `[Book Alg. 4.1]`.
- Python files use ready-made [Gymnasium](https://gymnasium.farama.org/) environments (FrozenLake, ...) or plain numpy when no environment is needed.
- Every file runs as-is with its default settings. Outputs (plots, saved tables) go to a `results/` folder next to the file and are not tracked by Git.
