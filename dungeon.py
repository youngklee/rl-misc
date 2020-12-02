import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class DungeonEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.dungeon_len = 5
        self.max_steps = 20
        self.large_reward = 20
        self.small_reward = 1
        self.seed()
        self.state = None
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.dungeon_len,))
        self.action_space = spaces.Discrete(2)
        self.steps = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def state_to_obs(self, state):
        return np.array([1 if i == state else 0 for i in range(self.dungeon_len)])

    def step(self, action):
        self.steps += 1
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        if action == 1:
            if self.state < self.dungeon_len - 2:
                self.state += 1
                reward = 0
            elif self.state == self.dungeon_len - 2:
                self.state += 1
                reward = self.large_reward
            else:
                reward = self.large_reward
        else:
            self.state = 0
            reward = self.small_reward

        done = self.steps >= self.max_steps
        obs = self.state_to_obs(self.state)
        return obs, reward, done, {}

    def reset(self):
        self.steps = 0
        self.state = self.np_random.randint(self.dungeon_len)
        return self.state_to_obs(self.state)

    def render(self, mode='human'):
        pass

    def close(self):
        pass

def main():
    env = DungeonEnv()
    env.reset()
    max_steps = 10
    max_episodes = 10
    for episode in range(max_episodes):
        state = env.reset()
        print(f'Episode {episode} starts at initial state {state}')
        rewards = []

        for step in range(max_steps):
            # Pick an action
            action = np.random.randint(2)

            # Take the action
            new_state, reward, done, _ = env.step(action)

            print(new_state, action, reward)
            rewards.append(reward)

            if done:
                break

            state = new_state

if __name__ == '__main__':
    main()
