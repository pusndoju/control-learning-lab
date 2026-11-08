from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Robbins-Monro: find the root of g(w) = 0 using only noisy measurements g~(w) = g(w) + eta
# (book: Zhao, Chapter 6, eq. 6-5, Theorem 6.1, Fig. 6.3)
# Run: open this file in VS Code and press F5

RESULTS = Path(__file__).resolve().parent / "results"
NAME = 'ch06_robbins_monro_root_finding'


class RobbinsMonro:
    # Hyperparameters
    iterations = 500        # number of RM steps (Book Fig. 6.3 shows 50; more steps make the trend clearer)
    w1 = 0.0                # initial guess (Book Fig. 6.3 uses w1 = 0)
    noise_std = 1.0         # eta ~ N(0, noise_std^2): E[eta]=0, finite variance (Thm. 6.1 c)

    # Step size schedules a_k (k = 1, 2, 3, ...)
    SCHEDULES = {
        'a_k = 1/k':     lambda k: 1 / k,        # sum a_k = inf, sum a_k^2 < inf  -> satisfies Thm. 6.1 (b)
        'a_k = 0.2/k':   lambda k: 0.2 / k,      # also satisfies (b), smaller first steps
        'a_k = 0.1':     lambda k: 0.1,          # sum a_k^2 = inf  -> keeps fluctuating
        'a_k = 1/k^2':   lambda k: 1 / k ** 2,   # sum a_k < inf    -> may stop before reaching the root
    }
    diverge_limit = 1e3     # stop a run when |w| exceeds this (diverged)

    # One RM run  [Book eq. 6-5]: w_k+1 = w_k - a_k * g~(w_k, eta_k)
    def run(self, g, schedule, rng):
        w = self.w1
        history = [w]
        for k in range(1, self.iterations + 1):
            eta = rng.normal(0, self.noise_std)     # we only see g(w) + eta, never g itself
            w = w - schedule(k) * (g(w) + eta)
            if abs(w) > self.diverge_limit:             # diverged: fill the rest with NaN
                history += [np.nan] * (self.iterations + 1 - len(history))
                break
            history.append(w)
        return np.array(history)

    def train(self, seed=0):
        problems = {
            # Book example: g(w) = w^3 - 5, root 5^(1/3) = 1.71. g' is NOT bounded above (Thm. 6.1 a fails)
            'g(w) = w^3 - 5':      (lambda w: w ** 3 - 5, 5 ** (1 / 3)),
            # Book p.106: g(w) = tanh(w - 1) satisfies 0 < c1 <= g'(w) <= c2 near the root
            'g(w) = tanh(w - 1)':  (lambda w: np.tanh(w - 1), 1.0),
        }

        RESULTS.mkdir(exist_ok=True)
        plt.figure(figsize=(11, 4))
        for i, (title, (g, root)) in enumerate(problems.items()):
            plt.subplot(1, 2, i + 1)
            for label, schedule in self.SCHEDULES.items():
                rng = np.random.default_rng(seed)       # same noise for every schedule (fair comparison)
                w = self.run(g, schedule, rng)
                plt.plot(w, label=label)
                result = 'DIVERGED' if np.isnan(w[-1]) else f'w_final = {w[-1]:+.3f}'
                print(f'{title:20s} {label:12s}: {result}   (true root {root:.3f})')
            plt.axhline(root, color='k', ls='--', lw=1, label='true root')
            plt.ylim(-1, 4)
            plt.xlabel('k'); plt.ylabel('w_k'); plt.title(title); plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS / f'{NAME}.png')
        plt.show()


if __name__ == '__main__':
    rm = RobbinsMonro()
    rm.train(seed=0)        # change the seed to see another noise realization