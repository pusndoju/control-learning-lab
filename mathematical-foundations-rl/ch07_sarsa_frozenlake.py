import pickle
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Sarsa: TD learning of action values + epsilon-greedy policy improvement at every step
# (book: Zhao, Chapter 7, eq. 7-12, Algorithm 7.1)
# Run: open this file in VS Code and press F5

RESULTS = Path(__file__).resolve().parent / "results"
NAME = 'ch07_sarsa_frozenlake'


class Sarsa:
    # Hyperparameters
    gamma = 0.9                 # discount rate
    alpha = 0.1                 # constant learning rate alpha_t(s,a) = alpha  [Book Alg. 7.1]
    epsilon = 0.1               # epsilon of the epsilon-greedy policy

    ACTIONS = ('L', 'D', 'R', 'U')

    # epsilon-greedy action from q(s,.)  [Book Alg. 7.1, policy update]
    def act(self, q_s, rng):
        if rng.random() < self.epsilon:
            return int(rng.integers(len(q_s)))                          # explore
        return int(rng.choice(np.flatnonzero(q_s == q_s.max())))       # greedy, random tie-break

    def train(self, episodes, render=False, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
                       render_mode='human' if render else None)
        nS, nA = env.observation_space.n, env.action_space.n
        rng = np.random.default_rng()

        q = np.zeros((nS, nA))                      # q_0(s,a) = 0
        rewards = np.zeros(episodes)
        lengths = np.zeros(episodes)

        for i in range(episodes):
            s = env.reset()[0]
            a = self.act(q[s], rng)                 # a_0 from pi_0(s_0)
            terminated = truncated = False
            steps = 0
            while not terminated and not truncated:
                # Experience sample (r_t+1, s_t+1, a_t+1): a_t+1 follows the CURRENT policy -> on-policy
                s2, r, terminated, truncated, _ = env.step(a)
                a2 = self.act(q[s2], rng)
                # [Book eq. 7-12]: q(s,a) <- q(s,a) - alpha * [q(s,a) - (r + gamma * q(s',a'))]
                td_target = r + self.gamma * q[s2, a2] * (not terminated)
                q[s, a] = q[s, a] - self.alpha * (q[s, a] - td_target)
                # Policy update is implicit: act() always reads the newest q (epsilon-greedy)
                s, a = s2, a2
                steps += 1
            rewards[i] = r
            lengths[i] = steps
        env.close()

        print(f'Sarsa: {episodes} episodes, alpha={self.alpha}, epsilon={self.epsilon}')
        print('greedy policy:'); self.print_policy(q.argmax(1), env)

        RESULTS.mkdir(exist_ok=True)
        w = 100
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.convolve(rewards, np.ones(w) / w, mode='valid'))
        plt.xlabel('episode'); plt.title(f'Success rate (last {w})  [Fig. 7.2]')
        plt.subplot(1, 2, 2)
        plt.plot(np.convolve(lengths, np.ones(w) / w, mode='valid'))
        plt.xlabel('episode'); plt.title(f'Episode length (last {w})')
        plt.tight_layout()
        plt.savefig(RESULTS / f'{NAME}.png')
        with open(RESULTS / f'{NAME}.pkl', 'wb') as f:
            pickle.dump(q, f)

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
        print(f'Test (greedy): goal reached {wins}/{episodes}')

    def print_policy(self, policy, env):
        terminal = np.isin(env.unwrapped.desc.flatten(), [b'H', b'G'])
        for r in range(4):
            print('  ' + ' '.join('.' if terminal[4 * r + c] else self.ACTIONS[policy[4 * r + c]]
                                  for c in range(4)))


if __name__ == '__main__':
    sarsa = Sarsa()
    is_slippery = False

    sarsa.train(2000, render=False, is_slippery=is_slippery)
    sarsa.test(episodes=1000, render=False, is_slippery=is_slippery)
    sarsa.test(episodes=3, render=True, is_slippery=is_slippery)
    plt.show()