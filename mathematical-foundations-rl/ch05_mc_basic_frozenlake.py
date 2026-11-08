import pickle
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# MC Basic: policy iteration where q_pi_k(s,a) is estimated by averaging episode returns
# (book: Zhao, Chapter 5, Algorithm 5.1)
# Run: open this file in VS Code and press F5

RESULTS = Path(__file__).resolve().parent / "results"
NAME = 'ch05_mc_basic_frozenlake'


class MCBasic:
    # Hyperparameters
    gamma = 0.9                 # discount rate
    episodes_per_pair = 1       # episodes collected from each (s,a); 1 is enough if env and policy are deterministic
    episode_length = 20         # max steps per episode; too short -> wrong q (Book Fig. 5.4)
    iterations = 10             # number of policy iterations k

    ACTIONS = ('L', 'D', 'R', 'U')

    # Generate one episode that starts from (s, a), then follows policy, return its discounted return
    def episode_return(self, env, s, a, policy):
        env.reset()
        env.unwrapped.s = s                         # force the start state (needed for "start from (s,a)")
        g, discount = 0.0, 1.0
        action = a
        for t in range(self.episode_length):
            s, r, terminated, truncated, _ = env.step(action)
            g += discount * r                       # G = r1 + gamma*r2 + gamma^2*r3 + ...  [Book eq. 5-2]
            discount *= self.gamma
            if terminated or truncated:
                break
            action = policy[s]                      # after the first step, follow pi_k
        return g

    def train(self, render=False, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
                       render_mode='human' if render else None)
        nS, nA = env.observation_space.n, env.action_space.n
        desc = env.unwrapped.desc.flatten()
        terminal = np.isin(desc, [b'H', b'G'])      # holes and goal: no decision needed

        v_star = self.true_optimal_value(env)       # only used to check the result
        policy = np.zeros(nS, dtype=int)            # pi_0: always 'L'
        q = np.zeros((nS, nA))
        errors = []

        for k in range(self.iterations):
            for s in range(nS):
                if terminal[s]:
                    continue
                for a in range(nA):
                    # Policy evaluation: q_k(s,a) = average return of episodes starting from (s,a)  [Book Alg. 5.1]
                    returns = [self.episode_return(env, s, a, policy) for _ in range(self.episodes_per_pair)]
                    q[s, a] = np.mean(returns)
                # Policy improvement: a*_k(s) = argmax_a q_k(s,a)  [Book Alg. 5.1]
                policy[s] = q[s].argmax()

            errors.append(np.max(np.abs(q.max(1) - v_star)))
            print(f'k={k}: max |max_a q_k - v*| = {errors[-1]:.4f}')
        env.close()

        print('max_a q(s,a):'); self.print_grid(q.max(1))
        print('policy:'); self.print_policy(policy, terminal)

        RESULTS.mkdir(exist_ok=True)
        plt.figure(figsize=(5, 4))
        plt.plot(errors, 'o-')
        plt.xlabel('iteration k'); plt.ylabel('max |max_a q_k - v*|')
        plt.title(f'MC Basic (episode length = {self.episode_length})')
        plt.savefig(RESULTS / f'{NAME}.png')
        with open(RESULTS / f'{NAME}.pkl', 'wb') as f:
            pickle.dump(q, f)

    # Value iteration with the model, ONLY to compare (MC Basic itself never uses the model)
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
    mc = MCBasic()
    is_slippery = False     # True: also set episodes_per_pair = 100 or more

    mc.train(render=False, is_slippery=is_slippery)
    mc.test(episodes=1000, render=False, is_slippery=is_slippery)
    mc.test(episodes=3, render=True, is_slippery=is_slippery)