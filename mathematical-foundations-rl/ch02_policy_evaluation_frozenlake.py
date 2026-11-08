from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Policy evaluation: solve the Bellman equation v_pi = r_pi + gamma*P_pi*v_pi
# (book: Zhao, Mathematical Foundations of RL, Chapter 2, Sec. 2.7)
# Run: open this file in VS Code and press F5

RESULTS = Path(__file__).resolve().parent / "results"


class PolicyEvaluation:
    # Hyperparameters
    gamma = 0.9             # discount rate (book uses 0.9)
    tol = 1e-8              # stop iterating when ||v_k+1 - v_k||_inf < tol
    max_iter = 1000         # safety limit for the iterative solution

    ACTIONS = ('L', 'D', 'R', 'U')      # FrozenLake actions 0,1,2,3

    # Read the model p(s'|s,a) and p(r|s,a) from the environment
    def get_model(self, env):
        P = env.unwrapped.P                         # P[s][a] = [(prob, s', r, terminated), ...]
        nS, nA = env.observation_space.n, env.action_space.n
        p_next = np.zeros((nS, nA, nS))             # p(s'|s,a)
        r_mean = np.zeros((nS, nA))                 # sum_r p(r|s,a) * r  (expected immediate reward)
        for s in range(nS):
            for a in range(nA):
                for prob, s2, r, _ in P[s][a]:
                    p_next[s, a, s2] += prob
                    r_mean[s, a] += prob * r
        # Holes and goal loop to themselves with r = 0, so their value is 0 automatically
        return p_next, r_mean

    # Build r_pi and P_pi of a policy pi(a|s)  [Book eq. 2-9, 2-10]
    def policy_matrices(self, pi, p_next, r_mean):
        r_pi = np.sum(pi * r_mean, axis=1)                  # r_pi(s)    = sum_a pi(a|s) * r(s,a)
        P_pi = np.einsum('sa,sat->st', pi, p_next)          # P_pi(s,s') = sum_a pi(a|s) * p(s'|s,a)
        return r_pi, P_pi

    # Closed-form solution  [Book Sec. 2.7.1]: v_pi = (I - gamma*P_pi)^-1 r_pi
    def closed_form(self, r_pi, P_pi):
        n = len(r_pi)
        return np.linalg.solve(np.eye(n) - self.gamma * P_pi, r_pi)    # solve() is safer than inv()

    # Iterative solution  [Book eq. 2-11]: v_k+1 = r_pi + gamma*P_pi*v_k
    def iterative(self, r_pi, P_pi):
        v = np.zeros(len(r_pi))                     # v_0: any initial guess works [Book eq. 2-12]
        history = [v.copy()]
        for k in range(self.max_iter):
            v_new = r_pi + self.gamma * P_pi @ v    # [Book eq. 2-11]
            history.append(v_new.copy())
            if np.max(np.abs(v_new - v)) < self.tol:
                break
            v = v_new
        return v_new, np.array(history)

    # Action values from state values  [Book eq. 2-14]: q(s,a) = r(s,a) + gamma*sum_s' p(s'|s,a) v(s')
    def action_values(self, v, p_next, r_mean):
        return r_mean + self.gamma * p_next @ v

    # Evaluate one policy with both methods, print and plot
    def solve(self, pi, name, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery)
        p_next, r_mean = self.get_model(env)
        env.close()

        r_pi, P_pi = self.policy_matrices(pi, p_next, r_mean)
        v_closed = self.closed_form(r_pi, P_pi)
        v_iter, history = self.iterative(r_pi, P_pi)
        q = self.action_values(v_closed, p_next, r_mean)

        print(f'=== Policy: {name} (slippery={is_slippery}) ===')
        print('v_pi (closed form):'); self.print_grid(v_closed)
        print(f'Iterative solution: {len(history) - 1} iterations, '
              f'max |v_iter - v_closed| = {np.max(np.abs(v_iter - v_closed)):.2e}')
        print('q_pi(s,a) [Book eq. 2-14]:'); self.print_q(q)
        # [Book eq. 2-13]: v_pi(s) = sum_a pi(a|s) q_pi(s,a)  -> check consistency
        print(f'Check eq. 2-13: max |sum_a pi*q - v| = {np.max(np.abs(np.sum(pi * q, 1) - v_closed)):.2e}\n')

        # Error ||v_k - v_pi||_inf decays like gamma^k  [Book Box 2.1]
        err = np.max(np.abs(history - v_closed), axis=1)
        return v_closed, err

    # Run one episode with the policy so you can see it move
    def test(self, pi, episodes=3, render=True, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
                       render_mode='human' if render else None)
        rng = np.random.default_rng()
        for i in range(episodes):
            state = env.reset()[0]
            terminated = truncated = False
            while not terminated and not truncated:
                action = rng.choice(4, p=pi[state])         # sample a ~ pi(.|s)
                state, reward, terminated, truncated, _ = env.step(action)
            print(f'Test episode {i + 1}: {"goal" if reward == 1 else "failed"}')
        env.close()

    def print_grid(self, v):
        for row in v.reshape(4, 4):
            print('  ' + ' '.join(f'{x:6.3f}' for x in row))

    def print_q(self, q):
        for s in range(q.shape[0]):
            print(f'{s:02}:[' + ' '.join(f'{x:+.3f}' for x in q[s]) + ']', end='  ')
            if (s + 1) % 4 == 0:
                print()


if __name__ == '__main__':
    pe = PolicyEvaluation()
    is_slippery = False

    # Policy 1: uniform random, pi(a|s) = 1/4
    pi_random = np.full((16, 4), 0.25)

    # Policy 2: a "good" deterministic policy (0=L, 1=D, 2=R, 3=U), path 0-4-8-9-13-14-15
    good_actions = [1, 0, 1, 0,
                    1, 0, 1, 0,
                    2, 1, 1, 0,
                    0, 2, 2, 0]
    pi_good = np.eye(4)[good_actions]               # one-hot: pi(a|s) = 1 for the chosen action

    v_r, err_r = pe.solve(pi_random, 'random', is_slippery)
    v_g, err_g = pe.solve(pi_good, 'good', is_slippery)

    RESULTS.mkdir(exist_ok=True)
    plt.figure(figsize=(12, 4))
    for i, (v, title) in enumerate([(v_r, 'v_pi random'), (v_g, 'v_pi good')]):
        plt.subplot(1, 3, i + 1)
        plt.imshow(v.reshape(4, 4), cmap='viridis')
        for s in range(16):
            plt.text(s % 4, s // 4, f'{v[s]:.2f}', ha='center', va='center', color='w')
        plt.title(title)
    plt.subplot(1, 3, 3)
    plt.semilogy(err_r, label='random')
    plt.semilogy(err_g, label='good')
    plt.semilogy(err_g[0] * pe.gamma ** np.arange(len(err_g)), '--', label='gamma^k bound')
    plt.xlabel('k'); plt.title('||v_k - v_pi||_inf  [Box 2.1]'); plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / 'ch02_policy_evaluation_frozenlake.png')

    pe.test(pi_good, render=True, is_slippery=is_slippery)