import pickle
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# n-step Sarsa: target uses n real rewards, then bootstraps with q(s_t+n, a_t+n)
# n = 1 -> Sarsa, n = inf -> MC   (book: Zhao, Chapter 7, eq. 7-17)
# Run: open this file in VS Code and press F5

RESULTS = Path(__file__).resolve().parent / "results"
NAME = 'ch07_nstep_sarsa_frozenlake'


class NStepSarsa:
    # Hyperparameters
    gamma = 0.9                 # discount rate
    alpha = 0.1                 # constant learning rate
    epsilon = 0.1               # epsilon-greedy exploration
    n = 3                       # number of real rewards in the target

    ACTIONS = ('L', 'D', 'R', 'U')

    def act(self, q_s, rng):
        if rng.random() < self.epsilon:
            return int(rng.integers(len(q_s)))
        return int(rng.choice(np.flatnonzero(q_s == q_s.max())))

    def train(self, episodes, render=False, is_slippery=False, seed=None):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
                       render_mode='human' if render else None)
        nS, nA = env.observation_space.n, env.action_space.n
        rng = np.random.default_rng(seed)
        q = np.zeros((nS, nA))
        rewards = np.zeros(episodes)

        for i in range(episodes):
            s = env.reset(seed=int(rng.integers(1e9)))[0]
            a = self.act(q[s], rng)
            states, actions, rews = [s], [a], [0.0]     # rews[t+1] = r_t+1 (index 0 unused)
            T = np.inf                                  # time step when the episode ends
            t = 0
            while True:
                if t < T:
                    s2, r, terminated, truncated, _ = env.step(actions[t])
                    states.append(s2); rews.append(r)
                    if terminated or truncated:
                        T = t + 1
                    else:
                        actions.append(self.act(q[s2], rng))    # a_t+1 from current eps-greedy policy

                tau = t - self.n + 1                    # the time step whose q is updated now
                if tau >= 0:
                    # G = r_tau+1 + gamma*r_tau+2 + ... + gamma^(n-1)*r_tau+n   (stop early at T)
                    end = int(min(tau + self.n, T))
                    G = sum(self.gamma ** (k - tau - 1) * rews[k] for k in range(tau + 1, end + 1))
                    # + gamma^n * q(s_tau+n, a_tau+n) if the episode has not ended yet
                    if tau + self.n < T:
                        G += self.gamma ** self.n * q[states[tau + self.n], actions[tau + self.n]]
                    # [Book eq. 7-17]: q(s_tau,a_tau) <- q - alpha * [q - G]
                    s_tau, a_tau = states[tau], actions[tau]
                    q[s_tau, a_tau] -= self.alpha * (q[s_tau, a_tau] - G)

                if tau == T - 1:
                    break
                t += 1
            rewards[i] = rews[-1]
        env.close()
        return q, rewards

    def test(self, q, episodes=10, render=True, is_slippery=False):
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

    def print_policy(self, policy):
        terminal = [5, 7, 11, 12, 15]
        for r in range(4):
            print('  ' + ' '.join('.' if 4 * r + c in terminal else self.ACTIONS[policy[4 * r + c]]
                                  for c in range(4)))


if __name__ == '__main__':
    algo = NStepSarsa()
    is_slippery = False
    episodes = 1000
    runs = 20                   # average several seeds: one run is too noisy to compare n

    RESULTS.mkdir(exist_ok=True)
    plt.figure(figsize=(6, 4))
    for n in [1, 3, 10]:
        algo.n = n
        curves = []
        for seed in range(runs):
            q, rewards = algo.train(episodes, is_slippery=is_slippery, seed=seed)
            curves.append(rewards)
        mean_curve = np.mean(curves, axis=0)
        first = np.argmax(np.convolve(mean_curve, np.ones(20) / 20, mode='valid') > 0.8)
        print(f'n = {n:2d}: mean success rate reaches 80% around episode {first}')
        plt.plot(np.convolve(mean_curve, np.ones(20) / 20, mode='valid'), label=f'n = {n}')
    plt.xlabel('episode'); plt.ylabel(f'success rate (mean of {runs} runs)')
    plt.title('n-step Sarsa  [eq. 7-17]'); plt.legend()
    plt.savefig(RESULTS / f'{NAME}.png')

    print('greedy policy of the last run (n = 10):'); algo.print_policy(q.argmax(1))
    with open(RESULTS / f'{NAME}.pkl', 'wb') as f:
        pickle.dump(q, f)
    algo.test(q, episodes=1000, render=False, is_slippery=is_slippery)
    algo.test(q, episodes=3, render=True, is_slippery=is_slippery)
    plt.show()