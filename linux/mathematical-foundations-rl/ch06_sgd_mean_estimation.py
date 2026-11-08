from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# SGD, mini-batch GD and batch GD for mean estimation: min_w J(w) = 1/(2n) sum ||w - x_i||^2
# (book: Zhao, Chapter 6, Sec. 6.4.4, Fig. 6.5)
# Run: open this file in VS Code and press F5

RESULTS = Path(__file__).resolve().parent / "results"
NAME = 'ch06_sgd_mean_estimation'


class MeanEstimationGD:
    # Hyperparameters
    n_samples = 100             # n: number of samples x_i (Book Fig. 6.5 uses 100)
    side = 20.0                 # X uniform in a square centered at the origin with this side length
    iterations = 30             # number of gradient steps
    w1 = np.array([-20.0, 15.0])  # initial guess, far from the mean
    batch_sizes = [1, 5, 50]    # m = 1 is SGD; m > 1 is MBGD

    # Gradient of f(w, x) = ||w - x||^2 / 2 is (w - x)  [Book eq. 6-14]
    def grad(self, w, x):
        return w - x

    def train(self, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.uniform(-self.side / 2, self.side / 2, size=(self.n_samples, 2))
        x_bar = X.mean(axis=0)                      # w* = sample mean

        runs = {}
        # BGD: uses all n samples every step  [Book Sec. 6.4.4, BGD]
        w = self.w1.copy(); path = [w.copy()]
        for k in range(1, self.iterations + 1):
            alpha = 1 / k
            w = w - alpha * np.mean(self.grad(w, X), axis=0)
            path.append(w.copy())
        runs['BGD'] = np.array(path)

        # MBGD / SGD: m samples drawn at random (with replacement) every step  [Book eq. 6-13]
        for m in self.batch_sizes:
            w = self.w1.copy(); path = [w.copy()]
            for k in range(1, self.iterations + 1):
                alpha = 1 / k                           # alpha_k = 1/k: sum = inf, sum of squares < inf
                batch = X[rng.integers(self.n_samples, size=m)]
                w = w - alpha * np.mean(self.grad(w, batch), axis=0)
                path.append(w.copy())
            runs['SGD (m=1)' if m == 1 else f'MBGD (m={m})'] = np.array(path)

        for name, path in runs.items():
            print(f'{name:12s}: final distance to mean = {np.linalg.norm(path[-1] - x_bar):.3f}')
        # With alpha_k = 1/k, BGD gives w_2 = x_bar exactly after ONE step  [Book Sec. 6.4.4]
        print(f'BGD after one step: ||w_2 - x_bar|| = {np.linalg.norm(runs["BGD"][1] - x_bar):.2e}')

        RESULTS.mkdir(exist_ok=True)
        plt.figure(figsize=(11, 4.5))
        plt.subplot(1, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], s=8, c='lightgray', label='samples')
        for name, path in runs.items():
            plt.plot(path[:, 0], path[:, 1], '.-', label=name)
        plt.plot(*x_bar, 'k*', ms=14, label='mean')
        plt.axis('equal'); plt.legend(); plt.title('Estimates w_k')

        plt.subplot(1, 2, 2)
        for name, path in runs.items():
            plt.plot(np.linalg.norm(path - x_bar, axis=1), label=name)
        plt.xlabel('iteration k'); plt.ylabel('||w_k - mean||'); plt.legend()
        plt.title('Distance to mean  [Fig. 6.5]')
        plt.tight_layout()
        plt.savefig(RESULTS / f'{NAME}.png')
        plt.show()


if __name__ == '__main__':
    gd = MeanEstimationGD()
    gd.train(seed=0)