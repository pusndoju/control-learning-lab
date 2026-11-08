import pickle
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Q-learning: TD learning of OPTIMAL action values, target uses max_a q(s', a)
# off-policy (Book Alg. 7.3): data from a behavior policy, learn the greedy target policy
# on-policy  (Book Alg. 7.2): data from the eps-greedy policy of the current q
# (book: Zhao, Chapter 7, eq. 7-18, Fig. 7.4)
# Run: open this file in VS Code and press F5

RESULTS = Path(__file__).resolve().parent / "results"
NAME = 'ch07_q_learning_frozenlake'


class QLearning:
    # Hyperparameters
    gamma = 0.9                 # discount rate
    alpha = 0.1                 # constant learning rate (Book Fig. 7.4 uses 0.1)
    off_policy = True           # True = Alg. 7.3, False = Alg. 7.2
    behavior_epsilon = 1.0      # off-policy behavior: eps-greedy around q; 1.0 = uniform random (Book Fig. 7.4)
    epsilon = 0.1               # on-policy only: epsilon of the eps-greedy policy

    ACTIONS = ('L', 'D', 'R', 'U')

    def act(self, q_s, eps, rng):
        if rng.random() < eps:
            return int(rng.integers(len(q_s)))
        return int(rng.choice(np.flatnonzero(q_s == q_s.max())))

    # Train for a total number of environment steps (episodes restart at the start state)
    def train(self, total_steps, render=False, is_slippery=False, seed=None):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
                       render_mode='human' if render else None)
        nS, nA = env.observation_space.n, env.action_space.n
        rng = np.random.default_rng(seed)
        v_star = self.true_optimal_value(env)       # ground truth from the model, ONLY to measure error

        q = np.zeros((nS, nA))
        eps = self.behavior_epsilon if self.off_policy else self.epsilon
        rmse = []
        s = env.reset(seed=int(rng.integers(1e9)))[0]

        for step in range(total_steps):
            a = self.act(q[s], eps, rng)            # behavior policy (off) or current eps-greedy (on)
            s2, r, terminated, truncated, _ = env.step(a)
            # [Book eq. 7-18]: q(s,a) <- q(s,a) - alpha * [q(s,a) - (r + gamma * max_a' q(s',a'))]
            td_target = r + self.gamma * q[s2].max() * (not terminated)
            q[s, a] = q[s, a] - self.alpha * (q[s, a] - td_target)
            # Target policy = greedy w.r.t. q  [Book Alg. 7.3]  (implicit: argmax when testing)

            s = s2
            if terminated or truncated:
                s = env.reset()[0]
            if step % 100 == 0:
                rmse.append(np.sqrt(np.mean((q.max(1) - v_star) ** 2)))
        env.close()

        mode = 'off-policy' if self.off_policy else 'on-policy'
        print(f'Q-learning {mode}, eps={eps}: RMSE(max_a q, v*) = {rmse[-1]:.4f} after {total_steps} steps')
        return q, rmse

    # Value iteration with the model, ONLY as ground truth
    def true_optimal_value(self, env):
        P = env.unwrapped.P
        nS, nA = env.observation_space.n, env.action_space.n
        v = np.zeros(nS)
        for _ in range(1000):
            v = np.array([max(sum(p * (r + self.gamma * v[s2]) for p, s2, r, _ in P[s][a])
                              for a in range(nA)) for s in range(nS)])
        return v

    def test(self, episodes=10, render=True, is_slippery=False):
        with open(RESULTS / f'{NAME}.pkl', 'rb') as f:
            q = pickle.load(f)
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
                       render_mode='human' if render else None)
        wins = 0
        for i in range(episodes):
            state = env.reset()[0]
            terminated = truncated = False
            while not terminated and not truncated:
                state, reward, terminated, truncated, _ = env.step(int(q[state].argmax()))
            wins += int(reward == 1)
        env.close()
        print(f'Test (greedy target policy): goal reached {wins}/{episodes}')

    def print_policy(self, policy):
        terminal = [5, 7, 11, 12, 15]
        for r in range(4):
            print('  ' + ' '.join('.' if 4 * r + c in terminal else self.ACTIONS[policy[4 * r + c]]
                                  for c in range(4)))


if __name__ == '__main__':
    ql = QLearning()
    is_slippery = False
    total_steps = 100_000       # Book Fig. 7.4: 100,000 steps of the behavior policy

    # Compare behavior policies: uniform (eps=1) vs less exploratory (Book Fig. 7.5)
    RESULTS.mkdir(exist_ok=True)
    plt.figure(figsize=(6, 4))
    for eps in [1.0, 0.5, 0.1]:
        ql.behavior_epsilon = eps
        q, rmse = ql.train(total_steps, is_slippery=is_slippery, seed=0)
        plt.plot(np.arange(len(rmse)) * 100, rmse, label=f'behavior eps = {eps}')
        if eps == 1.0:
            q_uniform = q
    plt.xlabel('step'); plt.ylabel('RMSE of max_a q vs v*')
    plt.title('Off-policy Q-learning  [Fig. 7.4, 7.5]'); plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / f'{NAME}.png')

    print('target policy learned from uniform behavior data:'); ql.print_policy(q_uniform.argmax(1))
    with open(RESULTS / f'{NAME}.pkl', 'wb') as f:
        pickle.dump(q_uniform, f)
    ql.test(episodes=1000, render=False, is_slippery=is_slippery)
    ql.test(episodes=3, render=True, is_slippery=is_slippery)
    plt.show()