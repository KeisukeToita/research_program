#Training全体を受け持つクラス

from agent import *
from maze import *

class Trainer():

    def __init__(self, agent, env, episode=1):
        self.env = env
        self.agent = agent

        self.episode = episode

    def train(self):
        for i in range(self.episode):
            print("Episode {}: Agent gets {} reward.".format(i, self.one_episode()))

    def one_episode(self):
        state = self.env.reset()
        total_reward = 0
        done = False

        while not done:
            action = self.agent.act(state)
            next_state, reward, done = self.env.step(action)
            total_reward += reward
            state = next_state

        return total_reward 

    def learn(self):
        pass

    def log(self):
        pass


def main():
    #環境データ
     grid = [
         [0,0,0,1],
         [0,9,0,-1],
         [0,0,0,0]
     ]

     env = Maze(grid)
     agent = Agent(env)

     trainer = Trainer(agent, env)

     trainer.train()

if __name__=="__main__":
    main()


    