from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# TD(0) learning of state values of a GIVEN policy (policy evaluation, model-free)
# (book: Zhao, Chapter 7, eq. 7-1, Theorem 7.1)
# Run: open this file in VS Code and press F5

RESULTS = Path(__file__).resolve().parent / "results"
NAME = 'ch07_td_state_value_frozenlake'


class TDStateValue:
    # Hyperparameters
    gamma = 0.9                 # discount rate
    alpha_constant = 0.1        # constant step size (sum alpha^2 = inf: converges only in expectation)
    # alpha = 1/N(s): sum = inf and sum^2 < inf for every visited state -> satisfies Thm. 7.1

    # Policy to evaluate: uniform random, pi(a|s) = 1/4 (exploratory, so every state gets visited)
    def policy(self, state, rng):
        return int(rng.integers(4))

    # One TD run. alpha_mode = 'constant' or '1/N'
    def train(self, episodes, alpha_mode='constant', render=False, is_slippery=False, seed=0):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
                       render_mode='human' if render else None)
        nS = env.observation_space.n
        rng = np.random.default_rng(seed)
        v_true = self.true_value(env)               # model-based v_pi, ONLY to measure the error

        v = np.zeros(nS)                            # v_0(s) = 0; terminal states stay 0
        visits = np.zeros(nS)
        rmse = []

        for i in range(episodes):
            s = env.reset(seed=int(rng.integers(1e9)))[0]
            terminated = truncated = False
            while not terminated and not truncated:
                a = self.policy(s, rng)
                s2, r, terminated, truncated, _ = env.step(a)

                visits[s] += 1
                alpha = self.alpha_constant if alpha_mode == 'constant' else 1 / visits[s]
                # TD target  v_bar = r_t+1 + gamma * v_t(s_t+1)   (v(terminal) = 0)
                td_target = r + self.gamma * v[s2] * (not terminated)
                # TD error   delta = v_t(s_t) - v_bar
                td_error = v[s] - td_target
                # [Book eq. 7-1]: v_t+1(s_t) = v_t(s_t) - alpha * delta ; other states unchanged [eq. 7-2]
                v[s] = v[s] - alpha * td_error
                s = s2
            rmse.append(np.sqrt(np.mean((v - v_true) ** 2)))
        env.close()

        print(f'alpha = {alpha_mode}: RMSE after {episodes} episodes = {rmse[-1]:.4f}')
        return v, v_true, rmse

    # v_pi from the Bellman equation with the model [Book Sec. 2.7.1], ONLY for checking
    def true_value(self, env):
        P = env.unwrapped.P
        nS, nA = env.observation_space.n, env.action_space.n
        r_pi, P_pi = np.zeros(nS), np.zeros((nS, nS))
        for s in range(nS):
            for a in range(nA):
                for prob, s2, r, _ in P[s][a]:
                    r_pi[s] += 0.25 * prob * r
                    P_pi[s, s2] += 0.25 * prob
        return np.linalg.solve(np.eye(nS) - self.gamma * P_pi, r_pi)

    def print_grid(self, v):
        for row in v.reshape(4, 4):
            print('  ' + ' '.join(f'{x:6.3f}' for x in row))


if __name__ == '__main__':
    td = TDStateValue()
    is_slippery = False
    episodes = 5000

    v_c, v_true, rmse_c = td.train(episodes, 'constant', is_slippery=is_slippery)
    v_n, _, rmse_n = td.train(episodes, '1/N', is_slippery=is_slippery)

    print('true v_pi (random policy):'); td.print_grid(v_true)
    print('TD estimate, alpha = 1/N:'); td.print_grid(v_n)

    RESULTS.mkdir(exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(rmse_c, label=f'alpha = {td.alpha_constant}')
    plt.plot(rmse_n, label='alpha = 1/N(s)')
    plt.xlabel('episode'); plt.ylabel('RMSE ||v - v_pi||')
    plt.title('TD(0) policy evaluation  [eq. 7-1]'); plt.legend()
    plt.savefig(RESULTS / f'{NAME}.png')
    plt.show()