import pickle
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Policy iteration: policy evaluation (solve v_pi exactly) + policy improvement (greedy)
# (book: Zhao, Chapter 4, Algorithm 4.2, Theorem 4.1)
# Run: open this file in VS Code and press F5

RESULTS = Path(__file__).resolve().parent / "results"
NAME = 'ch04_policy_iteration_frozenlake'


class PolicyIteration:
    # Hyperparameters
    gamma = 0.9             # discount rate
    eval_tol = 1e-8         # inner loop: stop policy evaluation when ||v^(j+1) - v^(j)|| < eval_tol
    max_outer = 100         # safety limit for policy iterations

    ACTIONS = ('L', 'D', 'R', 'U')

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

    # Policy evaluation by iteration  [Book eq. 4-4]: v^(j+1) = r_pi + gamma*P_pi*v^(j)
    def evaluate(self, policy, p_next, r_mean):
        nS = len(policy)
        r_pi = r_mean[np.arange(nS), policy]            # deterministic policy: r_pi(s) = r(s, pi(s))
        P_pi = p_next[np.arange(nS), policy]            # P_pi(s,s') = p(s'|s, pi(s))
        v = np.zeros(nS)
        j = 0
        while True:
            v_new = r_pi + self.gamma * P_pi @ v
            j += 1
            if np.max(np.abs(v_new - v)) < self.eval_tol:
                return v_new, j
            v = v_new

    def solve(self, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery)
        p_next, r_mean = self.get_model(env)
        env.close()
        nS, nA = r_mean.shape

        policy = np.zeros(nS, dtype=int)                # pi_0: always 'L' (a bad initial policy)
        v_history = []
        for k in range(self.max_outer):
            # ---- Policy evaluation: v_pi_k  [Book eq. 4-3] ----
            v, inner = self.evaluate(policy, p_next, r_mean)
            v_history.append(v)

            # ---- Policy improvement: q_pi_k(s,a), then greedy  [Book eq. 4-5] ----
            q = r_mean + self.gamma * p_next @ v
            new_policy = q.argmax(axis=1)

            print(f'k={k}: inner evaluation iterations = {inner}, sum v_pi_k = {v.sum():.4f}')
            # Stop when the policy no longer changes (then v_pi_k = v*)
            if np.array_equal(new_policy, policy):
                break
            policy = new_policy

        # Thm. 4.1: v_pi_k+1 >= v_pi_k elementwise (policy improvement)
        V = np.array(v_history)
        print('Monotonic improvement v_pi_k+1 >= v_pi_k:', bool(np.all(np.diff(V, axis=0) >= -1e-12)))
        print('v* :'); self.print_grid(v)
        print('pi*:'); self.print_policy(policy)

        RESULTS.mkdir(exist_ok=True)
        plt.figure(figsize=(5, 4))
        plt.plot(V, 'o-')
        plt.xlabel('policy iteration k'); plt.ylabel('v_pi_k(s)  (one line per state)')
        plt.title('Policy iteration: values never decrease')
        plt.savefig(RESULTS / f'{NAME}.png')

        with open(RESULTS / f'{NAME}.pkl', 'wb') as f:
            pickle.dump({'v': v, 'policy': policy}, f)
        return v, policy

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
    pi_algo = PolicyIteration()
    is_slippery = False

    pi_algo.solve(is_slippery)
    pi_algo.test(episodes=3, render=True, is_slippery=is_slippery)