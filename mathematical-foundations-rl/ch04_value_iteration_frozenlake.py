import pickle
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Value iteration: solve the Bellman optimality equation v = max_pi (r_pi + gamma*P_pi*v)
# (book: Zhao, Chapter 3 Thm. 3.3 + Chapter 4, Algorithm 4.1)
# Run: open this file in VS Code and press F5

RESULTS = Path(__file__).resolve().parent / "results"
NAME = 'ch04_value_iteration_frozenlake'


class ValueIteration:
    # Hyperparameters
    gamma = 0.9             # discount rate
    tol = 1e-6              # stop when ||v_k+1 - v_k||_inf < tol
    max_iter = 1000         # safety limit

    ACTIONS = ('L', 'D', 'R', 'U')

    # Model p(s'|s,a) and expected reward r(s,a) = sum_r p(r|s,a) r
    def get_model(self, env):
        P = env.unwrapped.P
        nS, nA = env.observation_space.n, env.action_space.n
        p_next = np.zeros((nS, nA, nS))
        r_mean = np.zeros((nS, nA))
        for s in range(nS):
            for a in range(nA):
                for prob, s2, r, _ in P[s][a]:
                    p_next[s, a, s2] += prob
                    r_mean[s, a] += prob * r
        return p_next, r_mean

    def solve(self, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery)
        p_next, r_mean = self.get_model(env)
        env.close()
        nS, nA = r_mean.shape

        v = np.zeros(nS)                            # initial guess v_0 (any value converges, Thm. 3.3)
        deltas = []
        for k in range(self.max_iter):
            # q-value: q_k(s,a) = r(s,a) + gamma * sum_s' p(s'|s,a) v_k(s')   [Book Alg. 4.1]
            q = r_mean + self.gamma * p_next @ v
            # Policy update: greedy action a*_k(s) = argmax_a q_k(s,a)       [Book eq. 4-2]
            greedy = q.argmax(axis=1)
            # Value update: v_k+1(s) = max_a q_k(s,a)                         [Book eq. 4-1]
            v_new = q.max(axis=1)

            deltas.append(np.max(np.abs(v_new - v)))
            if deltas[-1] < self.tol:
                v = v_new
                break
            v = v_new

        print(f'Converged after {k + 1} iterations (slippery={is_slippery})')
        print('v* :'); self.print_grid(v)
        print('pi*:'); self.print_policy(greedy)

        # ||v_k+1 - v_k|| shrinks at least by gamma each step (contraction, Thm. 3.2)
        RESULTS.mkdir(exist_ok=True)
        plt.figure(figsize=(5, 4))
        plt.semilogy(deltas, 'o-', label='||v_k+1 - v_k||_inf')
        plt.semilogy(deltas[0] * self.gamma ** np.arange(len(deltas)), '--', label='gamma^k bound')
        plt.xlabel('k'); plt.legend(); plt.title('Value iteration')
        plt.savefig(RESULTS / f'{NAME}.png')

        with open(RESULTS / f'{NAME}.pkl', 'wb') as f:
            pickle.dump({'v': v, 'policy': greedy}, f)
        return v, greedy

    # Follow the greedy policy found by value iteration
    def test(self, episodes=3, render=True, is_slippery=False):
        with open(RESULTS / f'{NAME}.pkl', 'rb') as f:
            policy = pickle.load(f)['policy']
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
                       render_mode='human' if render else None)
        wins = 0
        for i in range(episodes):
            state = env.reset()[0]
            terminated = truncated = False
            while not terminated and not truncated:
                state, reward, terminated, truncated, _ = env.step(int(policy[state]))
            wins += int(reward == 1)
        env.close()
        print(f'Test: goal reached {wins}/{episodes}')

    def print_grid(self, v):
        for row in v.reshape(4, 4):
            print('  ' + ' '.join(f'{x:6.3f}' for x in row))

    def print_policy(self, policy):
        for row in np.array(policy).reshape(4, 4):
            print('  ' + ' '.join(self.ACTIONS[a] for a in row))


if __name__ == '__main__':
    vi = ValueIteration()
    is_slippery = False     # True = stochastic transitions (1/3 intended, 1/3 each side)

    vi.solve(is_slippery)
    vi.test(episodes=3, render=True, is_slippery=is_slippery)