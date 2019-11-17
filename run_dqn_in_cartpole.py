import gym
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from dqn_agent import DQNAgent
# from ddqn_agent import DDQNAgent
from policy import EpsGreedyQPolicy
from memory import Memory


def obs_processer(row_obs):
    return row_obs


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n
    actions = np.arange(nb_actions)
    policy = EpsGreedyQPolicy(eps=1., eps_decay_rate=.999, min_eps=.01)
    memory = Memory(limit=50000, maxlen=1)
    obs = env.reset()
    agent = DQNAgent(actions=actions,
                     memory=memory,
                     update_interval=200,
                     train_interval=1,
                     batch_size=32,
                     observation=obs,
                     input_shape=[len(obs)],
                     policy=policy,
                     obs_processer=obs_processer)

    agent.compile()

    result = []
    nb_epsiodes = 1000
    for episode in range(nb_epsiodes):
        agent.reset()
        observation = env.reset()
        observation = deepcopy(observation)
        agent.observe(observation)
        done = False
        while not done:
            action = deepcopy(agent.act())
            observation, reward, done, info = env.step(action)
            observation = deepcopy(observation)
            agent.observe(observation, reward, done)
            if done:
                break

        agent.training = False
        observation = env.reset()
        agent.observe(observation)
        done = False
        step = 0
        while not done:
            # env.render() # 表示
            step += 1
            action = agent.act()
            observation, reward, done, info = env.step(action)
            agent.observe(observation)
            if done:
                print("Episode {}: {} steps".format(episode, step))
                result.append(step)
                break

        agent.training = True

    x = np.arange(len(result))
    plt.ylabel("time")
    plt.xlabel("episode")
    plt.plot(x, result)
    plt.savefig("result.png")
