import pickle
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# MC epsilon-greedy: no exploring starts, the epsilon-greedy policy itself explores
# (book: Zhao, Chapter 5, Algorithm 5.3)
# Run: open this file in VS Code and press F5

RESULTS = Path(__file__).resolve().parent / "results"
NAME = 'ch05_mc_epsilon_greedy_frozenlake'


class MCEpsilonGreedy:
    # Hyperparameters
    gamma = 0.9                 # discount rate
    epsilon = 0.5               # fixed epsilon in (0,1]; large = more exploration, less optimal (Book Sec. 5.5)
    episode_length = 100        # T: max steps per episode

    ACTIONS = ('L', 'D', 'R', 'U')

    # [Book Alg. 5.3]: pi(a*|s) = 1 - eps*(|A|-1)/|A|,  pi(a|s) = eps/|A| for a != a*
    def epsilon_greedy(self, q_s, rng):
        nA = len(q_s)
        best = rng.choice(np.flatnonzero(q_s == q_s.max()))     # random tie-break
        probs = np.full(nA, self.epsilon / nA)
        probs[best] = 1 - self.epsilon * (nA - 1) / nA
        return probs

    def train(self, episodes, render=False, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
                       render_mode='human' if render else None)
        nS, nA = env.observation_space.n, env.action_space.n
        terminal = np.isin(env.unwrapped.desc.flatten(), [b'H', b'G'])
        rng = np.random.default_rng()

        q = np.zeros((nS, nA))
        returns_sum = np.zeros((nS, nA))
        num = np.zeros((nS, nA))
        rewards = np.zeros(episodes)

        for i in range(episodes):
            # Start from the normal start state 0 (exploring starts NOT required)
            s = env.reset()[0]
            episode = []
            for t in range(self.episode_length):
                a = rng.choice(nA, p=self.epsilon_greedy(q[s], rng))   # follow current eps-greedy policy
                s2, r, terminated, truncated, _ = env.step(a)
                episode.append((s, a, r))
                s = s2
                if terminated or truncated:
                    break
            rewards[i] = r

            # Every-visit MC, backward  [Book Alg. 5.3]
            g = 0.0
            for t in reversed(range(len(episode))):
                st, at, r = episode[t]
                g = self.gamma * g + r
                returns_sum[st, at] += g
                num[st, at] += 1
                q[st, at] = returns_sum[st, at] / num[st, at]
                # Policy improvement is implicit: the next action is drawn from epsilon_greedy(q)
        env.close()

        greedy = q.argmax(1)
        print(f'epsilon = {self.epsilon}, {episodes} episodes')
        print('max_a q(s,a)  (value of the eps-greedy policy, not v*):'); self.print_grid(q.max(1))
        print('greedy policy:'); self.print_policy(greedy, terminal)

        RESULTS.mkdir(exist_ok=True)
        rate = np.convolve(rewards, np.ones(100) / 100, mode='valid')
        plt.figure(figsize=(5, 4))
        plt.plot(rate)
        plt.xlabel('episode'); plt.ylabel('success rate (last 100)')
        plt.title(f'MC eps-greedy, eps={self.epsilon} (training)')
        plt.savefig(RESULTS / f'{NAME}.png')
        with open(RESULTS / f'{NAME}.pkl', 'wb') as f:
            pickle.dump(q, f)

    # Test with the GREEDY policy derived from q (epsilon = 0)
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

    def print_grid(self, v):
        for row in v.reshape(4, 4):
            print('  ' + ' '.join(f'{x:6.3f}' for x in row))

    def print_policy(self, policy, terminal):
        for r in range(4):
            print('  ' + ' '.join('.' if terminal[4 * r + c] else self.ACTIONS[policy[4 * r + c]]
                                  for c in range(4)))


if __name__ == '__main__':
    mc = MCEpsilonGreedy()
    is_slippery = False

    mc.train(2000, render=False, is_slippery=is_slippery)
    mc.test(episodes=1000, render=False, is_slippery=is_slippery)
    mc.test(episodes=3, render=True, is_slippery=is_slippery)