import pickle
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Truncated policy iteration: policy evaluation stopped after j_truncate sweeps
# j_truncate = 1 -> value iteration, j_truncate = inf -> policy iteration
# (book: Zhao, Chapter 4, Algorithm 4.3, Fig. 4.5)
# Run: open this file in VS Code and press F5

RESULTS = Path(__file__).resolve().parent / "results"
NAME = 'ch04_truncated_policy_iteration_frozenlake'


class TruncatedPolicyIteration:
    # Hyperparameters
    gamma = 0.9             # discount rate
    tol = 1e-6              # outer loop: stop when ||v_k - v*||_inf < tol
    max_outer = 500         # safety limit

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

    # Exact v* (only used to measure the error of each run)
    def optimal_value(self, p_next, r_mean):
        v = np.zeros(r_mean.shape[0])
        while True:
            v_new = (r_mean + self.gamma * p_next @ v).max(axis=1)
            if np.max(np.abs(v_new - v)) < 1e-12:
                return v_new
            v = v_new

    # One run of Algorithm 4.3 with a given j_truncate
    def run(self, j_truncate, p_next, r_mean, v_star):
        nS = r_mean.shape[0]
        v = np.zeros(nS)                                # v_0
        q = r_mean + self.gamma * p_next @ v
        policy = q.argmax(axis=1)                       # pi_1 greedy w.r.t. v_0 (same start for all runs)
        errors = [np.max(np.abs(v - v_star))]

        for k in range(self.max_outer):
            # ---- Policy evaluation, truncated: start from v_k-1, do j_truncate sweeps ----
            r_pi = r_mean[np.arange(nS), policy]
            P_pi = p_next[np.arange(nS), policy]
            for j in range(j_truncate):
                v = r_pi + self.gamma * P_pi @ v        # [Book Alg. 4.3]: v^(j+1) = r_pi + gamma*P_pi*v^(j)

            # ---- Policy improvement ----
            q = r_mean + self.gamma * p_next @ v
            policy = q.argmax(axis=1)

            errors.append(np.max(np.abs(v - v_star)))
            if errors[-1] < self.tol:
                break
        return v, policy, errors

    def solve(self, j_list, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery)
        p_next, r_mean = self.get_model(env)
        env.close()
        v_star = self.optimal_value(p_next, r_mean)

        RESULTS.mkdir(exist_ok=True)
        plt.figure(figsize=(6, 4))
        for j in j_list:
            v, policy, errors = self.run(j, p_next, r_mean, v_star)
            label = 'value iteration' if j == 1 else f'j_truncate={j}'
            print(f'{label:>16}: {len(errors) - 1} outer iterations, '
                  f'{(len(errors) - 1) * j} evaluation sweeps in total')
            plt.semilogy(errors, 'o-', ms=3, label=label)
        plt.xlabel('outer iteration k'); plt.ylabel('||v_k - v*||_inf')
        plt.title('Truncated policy iteration  [Fig. 4.5]'); plt.legend()
        plt.savefig(RESULTS / f'{NAME}.png')

        print('pi*:'); self.print_policy(policy)
        with open(RESULTS / f'{NAME}.pkl', 'wb') as f:
            pickle.dump({'v': v, 'policy': policy}, f)

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

    def print_policy(self, policy):
        for row in np.array(policy).reshape(4, 4):
            print('  ' + ' '.join(self.ACTIONS[a] for a in row))


if __name__ == '__main__':
    tpi = TruncatedPolicyIteration()
    is_slippery = True      # slippery makes the difference between j values easier to see

    tpi.solve(j_list=[1, 3, 10, 100], is_slippery=is_slippery)
    tpi.test(episodes=2000, render=False, is_slippery=is_slippery)   # measure success rate
    tpi.test(episodes=10, render=True, is_slippery=is_slippery)      # watch 10 episodes