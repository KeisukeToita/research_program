# Training全体を受け持つクラス
from agent import *
from maze import *

class Trainer():

    def __init__(self, agent, env, episode=1, report_interval=50):
        self.env = env
        self.agent = agent

        self.episode = episode
        self.report_interval = report_interval

    def train(self):
        for i in range(self.episode):
            if (i+1) % self.report_interval == 0:
                print("Episode {}: Agent gets {} reward.".format(i+1, self.one_episode()))

    def one_episode(self):
        agent_state = self.env.reset()
        total_reward = 0
        done = False

        while not done:
            action = self.agent.act(agent_state)
            next_state, reward, done = self.env.step(action)
            total_reward += reward
            agent_state = next_state

        return total_reward


class MonteCarloTrainer(Trainer):
    def __init__(self, agent, env, episode=1, report_interval=50):
        super().__init__(agent, env, episode, report_interval)

    def one_episode(self):
        agent_state = self.env.reset()
        total_reward = 0
        done = False

        while not done:
            self.agent.init_log()
            while not done:
                a = self.agent.act(agent_state)
                n_state, reward, done = self.env.step(a)
                self.agent.experience_add( agent_state, a, reward)
                total_reward += reward
                agent_state = n_state
            else:
                self.agent.reward_add(reward)

        self.agent.learn()
        return total_reward

class QLearningTrainer(Trainer):
    def __init__(self, agent, env, episode=1, report_interval=50):
        super().__init__(agent, env, episode, report_interval)

    def one_episode(self):
        agent_state = self.env.reset()
        total_reward = 0
        done = False

        while not done:
            while not done:
                a = self.agent.act(agent_state)
                n_state, reward, done = self.env.step(a)
                self.agent.learn(agent_state, n_state, trans_aton(a), reward)
                total_reward += reward
                agent_state = n_state
            else:
                self.agent.reward_add(reward)

        return total_reward

def main():
    # 環境データ
    grid = [
        [0, 0, 0, 1],
        [0, 0, -1, -1],
        [0, 9, -1, 0],
        [0, 0, 0, 0],
    ]

    env = Maze(grid)
    agent = QLearningAgent(env, epsilon=0.2)

    trainer = QLearningTrainer(agent, env, 10000, 100)

    trainer.train()


if __name__ == "__main__":
    main()
