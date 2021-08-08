# Training全体を受け持つTrainerクラスを集めたファイル
from agent import *
from maze import *
from logger import *

#Single agent Trainer
class Trainer():

    def __init__(self, agents, env, episode=1, report_interval=50, dirname=None):
        self.env = env
        self.agents = agents #listにした

        self.episode = episode
        self.report_interval = report_interval

        self.logger = Logger(dirname, len(agents)) #Loggerにエージェントの数を入力

    def train(self):
        for i in range(self.episode):
            if (i+1) % self.report_interval == 0:
                print("Episode {}: Agent gets {} reward.".format(i+1, self.one_episode(i+1)))

    def one_episode(self, episode_n):
        agents_state=[]
        agents_done=[]
        for i in range(len(self.agents)):
            agents_state.append(self.env.reset())
            agents_done.append(False)

        total_reward = 0

        self.logger.init_exp_log()
        step_n = 1

        while False in agents_done:
            actions, next_states, rewards=[],[],[]
            
            #行動選択
            for i in range(len(self.agents)):
                actions.append(self.agents[i].act(agents_state[i]))
            
            #各エージェントごとにステップを行い，結果を格納
            for i in range(len(self.agents)):
                if agents_done[i] == False:
                    next_state, reward, done = self.env.step(i,actions[i])
                else:
                    actions[i]=self.env.actions[4] #STAY
                    next_state, reward, done = agents_state[i], int(0), True

                next_states.append(next_state)
                rewards.append(reward)
                agents_done[i]=done
                
            total_reward += sum(rewards)

            self.logger.add_experience(step_n, agents_state, actions, rewards, total_reward)

            for i in range(len(self.agents)):
                agents_state[i] = next_states[i]
            step_n += 1

        #csv形式で保存
        if episode_n % 5 == 0:
            self.logger.state_transition_write_csv(episode_n)

        return total_reward


class MonteCarloTrainer(Trainer):
    def __init__(self, agent, env, episode=1, report_interval=50, dirname=None):
        super().__init__(agent, env, episode, report_interval, dirname)

    def one_episode(self):
        agent_state = self.env.reset()
        total_reward = 0
        done = False

        self.logger.init_exp_log()
        step_n = 1

        while not done:
            self.agent.init_log()
            while not done:
                a = self.agent.act(agent_state)
                n_state, reward, done = self.env.step(a)
                self.agent.experience_add(agent_state, a, reward)
                total_reward += reward
                agent_state = n_state
            else:
                self.agent.reward_add(reward)

        self.agent.learn()
        return total_reward

class QLearningTrainer(Trainer):
    def __init__(self, agent, env, episode=1, report_interval=50, dirname=None):
        super().__init__(agent, env, episode, report_interval, dirname)

    def one_episode(self, episode_n):
        agent_state = self.env.reset()
        total_reward = 0
        done = False
        self.logger.init_exp_log()

        step_n = 1

        while not done:
            while not done:
                a = self.agent.act(agent_state)
                n_state, reward, done = self.env.step(a)
                self.agent.learn(agent_state, n_state, a, reward)
                total_reward += reward

                #データ格納
                self.logger.add_experience(step_n, agent_state, a, reward, total_reward)

                step_n += 1
                agent_state = n_state
            else:
                self.agent.reward_add(reward)

        #csv形式で保存
        if episode_n % 5 == 0:
            self.logger.state_transition_write_csv(episode_n)

        return total_reward
