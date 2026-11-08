import pickle
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# MC Exploring Starts: random (s0, a0) per episode, every-visit returns, policy updated episode by episode
# (book: Zhao, Chapter 5, Algorithm 5.2)
# Run: open this file in VS Code and press F5

RESULTS = Path(__file__).resolve().parent / "results"
NAME = 'ch05_mc_exploring_starts_frozenlake'


class MCExploringStarts:
    # Hyperparameters
    gamma = 0.9                 # discount rate
    episode_length = 30         # T: max steps per episode
    first_visit = False         # False = every-visit (Book Alg. 5.2), True = first-visit (Book Sec. 5.3.1)

    ACTIONS = ('L', 'D', 'R', 'U')

    def train(self, episodes, render=False, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
                       render_mode='human' if render else None)
        nS, nA = env.observation_space.n, env.action_space.n
        terminal = np.isin(env.unwrapped.desc.flatten(), [b'H', b'G'])
        starts = [(s, a) for s in range(nS) if not terminal[s] for a in range(nA)]
        rng = np.random.default_rng()

        q = np.zeros((nS, nA))
        returns_sum = np.zeros((nS, nA))            # Returns(s,a)
        num = np.zeros((nS, nA))                    # Num(s,a)
        policy = np.zeros(nS, dtype=int)            # pi_0: always 'L'
        v_star = self.true_optimal_value(env)
        errors = []

        for i in range(episodes):
            # Exploring starts: every (s,a) can be chosen as the start  [Book Alg. 5.2]
            s, a = starts[rng.integers(len(starts))]
            env.reset()
            env.unwrapped.s = s

            # Episode generation: s0,a0,r1, ..., s_T-1,a_T-1,r_T following the current policy
            episode = []
            for t in range(self.episode_length):
                s2, r, terminated, truncated, _ = env.step(a)
                episode.append((s, a, r))
                if terminated or truncated:
                    break
                s, a = s2, policy[s2]

            first_t = {}
            for t, (st, at, _) in enumerate(episode):
                first_t.setdefault((st, at), t)

            # Walk backward: g <- gamma*g + r_t+1  [Book Alg. 5.2]
            g = 0.0
            for t in reversed(range(len(episode))):
                st, at, r = episode[t]
                g = self.gamma * g + r
                if self.first_visit and first_t[(st, at)] != t:
                    continue
                returns_sum[st, at] += g
                num[st, at] += 1
                q[st, at] = returns_sum[st, at] / num[st, at]      # policy evaluation
                policy[st] = q[st].argmax()                        # policy improvement (greedy)

            errors.append(np.max(np.abs(q.max(1) - v_star)))
        env.close()

        print(f'After {episodes} episodes: max |max_a q - v*| = {errors[-1]:.4f}')
        print('max_a q(s,a):'); self.print_grid(q.max(1))
        print('policy:'); self.print_policy(policy, terminal)
        print('Visits Num(s,a) (min over non-terminal pairs):', int(num[~terminal].min()))

        RESULTS.mkdir(exist_ok=True)
        plt.figure(figsize=(5, 4))
        plt.plot(errors)
        plt.xlabel('episode'); plt.ylabel('max |max_a q - v*|')
        plt.title('MC Exploring Starts')
        plt.savefig(RESULTS / f'{NAME}.png')
        with open(RESULTS / f'{NAME}.pkl', 'wb') as f:
            pickle.dump(q, f)

    # Value iteration with the model, ONLY to compare
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
        print(f'Test: goal reached {wins}/{episodes}')

    def print_grid(self, v):
        for row in v.reshape(4, 4):
            print('  ' + ' '.join(f'{x:6.3f}' for x in row))

    def print_policy(self, policy, terminal):
        for r in range(4):
            print('  ' + ' '.join('.' if terminal[4 * r + c] else self.ACTIONS[policy[4 * r + c]]
                                  for c in range(4)))


if __name__ == '__main__':
    mc = MCExploringStarts()
    is_slippery = False

    mc.train(2000, render=False, is_slippery=is_slippery)
    mc.test(episodes=1000, render=False, is_slippery=is_slippery)
    mc.test(episodes=3, render=True, is_slippery=is_slippery)