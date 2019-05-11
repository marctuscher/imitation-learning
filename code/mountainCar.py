#%%
import itertools
import numpy as np
import gym
np.random.seed(0)
env = gym.make('MountainCar-v0')
env.seed(0)

#%%
class Agent:
    def decide(self, observation):
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,
                0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2 # push right
        else:
            action = 0 # push left
        return action

#%%
agent = Agent()
def play_once(env, agent, render=False, verbose=False):
    observation = env.reset()
    episode_reward = 0.
    for step in itertools.count():
        if render:
            env.render()
        action = agent.decide(observation)
        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if done:
            break
    if verbose:
        print('get {} rewards in {} steps'.format(
                episode_reward, step + 1))
    return episode_reward

#%%
episode_rewards = [play_once(env, agent, render=True) for _ in range(10)]
print('average episode rewards = {}'.format(np.mean(episode_rewards)))



#%%
