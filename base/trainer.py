# Training全体を受け持つTrainerクラスを集めたファイル
from matplotlib.pyplot import step
from agent import *
from maze import *
from logger import *
import numpy as np
import copy

np.random.seed(0)

#Single agent Trainer
class Trainer():

    def __init__(self, agents, env, episode=1, report_interval=50, dirname=None, seed=1):
        self.env = env
        self.agents = agents #listにした

        self.episode = episode
        self.report_interval = report_interval

        self.seed = seed

        #self.logger = Logger(dirname, len(agents)) #Loggerにエージェントの数を入力

    def train(self):
        for i in range(self.episode):
            if (i+1) % self.report_interval == 0:
                print("Episode {}: Agent gets {} reward.".format(i+1, self.one_episode(i+1)))

    def one_episode(self, episode_n):
        agents_state=[]
        agents_done=[]
        for i in range(len(self.agents)):
            agents_state.append(self.env.reset()[i])#TODO 初期位置リストを返す
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
                    next_state, reward, done = self.env.final_agents_state[i], int(0), True

                next_states.append(next_state)
                rewards.append(reward)
                agents_done[i]=done

            #衝突判定
            rewards, agents_done = self.env.colision_judge(next_states, rewards, agents_done)
            
            total_reward += sum(rewards)

            self.logger.add_experience(step_n, agents_state, actions, rewards, total_reward)

            for i in range(len(self.agents)):
                agents_state[i] = next_states[i]
            step_n += 1

        #csv形式で保存
        if episode_n % self.report_interval == 0:
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

class QLearningTrainer(Trainer):#マルチ用に改変
    def __init__(self, agents, env, episode=1, report_interval=50, dirname=None):
        print("~~~ Q_Learning_START ~~~")
        super().__init__(agents, env, episode, report_interval, dirname)

    def train(self):
        for i in range(self.episode):
            if (i+1) % self.report_interval == 0:
                reward, step = self.one_episode(i+1)
                print("Episode {}: Agent gets {} reward. {} step".format(i+1, reward, step))

    def one_episode(self, episode_n):
        agents_state=[]
        agents_done=[]
        for i in range(len(self.agents)):
            agents_state.append(self.env.reset()[i])
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
                    next_state, reward, done = self.env.get_finish_state(i), int(0), True

                next_states.append(next_state)
                rewards.append(reward)
                agents_done[i]=done

            #衝突判定
            rewards, agents_done = self.env.colision_judge(next_states, rewards, agents_done)

            for i in range(len(self.agents)):
                self.agents[i].learn(agents_state[i], next_states[i], actions[i], rewards[i])
            
            total_reward += sum(rewards)

            self.logger.add_experience(step_n, agents_state, actions, rewards, total_reward)

            for i in range(len(self.agents)):
                agents_state[i] = next_states[i]
            step_n += 1

        #csv形式で保存
        if episode_n % self.report_interval == 0:
            self.logger.state_transition_write_csv(episode_n)

        return total_reward, step_n

class SOMQLearningTrainer(Trainer):
    def __init__(self, agents, env, episode=1, report_interval=50, dirname=None, seed=1):
        print("~~~ Q_Learning_START ~~~")
        super().__init__(agents, env, episode, report_interval, dirname, seed)

        self.logger = Logger(dirname, len(agents))

        self.estimate_rates=np.zeros(len(self.agents))
        self.estimate_counts=np.zeros(len(self.agents))

        self.all_estimate_rates=[]

    def reset(self):
        self.estimate_rates=np.zeros(len(self.agents))
        self.estimate_counts=np.zeros(len(self.agents))
        for i in range(len(self.agents)):
            self.agents[i].seed_reset()

    def all_seed_train(self):
        for i in range(self.seed):
            print("{} seed train".format(i+1))
            self.reset()
            self.train()

        self.logger.show_ave_estimate_rate()
        self.logger.show_ave_estimate_rate_400_ave()
        

    def train(self):
        #train loop
        self.logger.seed_count()

        for i in range(self.episode):
            if (i+1) % self.report_interval == 0:
                reward, step = self.one_episode(i+1)
                print("Episode {}: Agent gets {} reward. {} step".format(i+1, reward, step))
            else:
                self.one_episode(i+1)

        #estimate transition save
        self.logger.show_estimate_rate()

        #show estimate rate
        for i in range(len(self.agents)):
            self.estimate_rates[i] = self.estimate_counts[i]/self.episode
            print("Agent {} rate: {}".format(i, self.estimate_rates[i]))

    def one_episode(self, episode_n):
        #init state & goal
        agents_state=[]
        agents_done=[]
        agents_goal=[]

        agents_state = self.env.reset()
        for i in range(len(self.agents)):
            agents_done.append(False)
            self.agents[i].reset()
            agents_goal.append(self.agents[i].get_my_goal())

        #TODO change general process
        other_goals=[]
        other_goals.append(self.agents[1].get_my_goal())
        other_goals.append(self.agents[0].get_my_goal())

        #環境側に各エージェントの目的をセット
        self.env.set_agents_goal(agents_goal)

        total_reward = 0

        self.logger.init_exp_log()
        step_n = 1

        while False in agents_done:
            actions, next_states, rewards=[],[],[]
            
            #choice actions
            for i in range(len(self.agents)):
                actions.append(self.agents[i].greedy_act(agents_state[i]))
            
            #one step
            for i in range(len(self.agents)):
                if agents_done[i] == False:
                    next_state, reward, done = self.env.step(i,actions[i])
                else:
                    actions[i]=self.env.actions[4] #STAY
                    next_state, reward, done = self.env.get_finish_state(i), int(0), True

                next_states.append(next_state)
                rewards.append(reward)
                agents_done[i]=done
            
            #is_colision??? 
            #rewards, agents_done = self.env.colision_judge(next_states, rewards, agents_done)

            #TODO generate the process
            #make other_states & action 一旦入れ替えるでオケかな
         
            other_states=[agents_state[1], agents_state[0]]
            other_actions=[actions[1], actions[0]]            
            
            #learn
            for i in range(len(self.agents)):
                self.agents[i].learn(agents_state[i], next_states[i], actions[i], rewards[i])
           
            est_other_goals=[]
            #estimate
            for i in range(len(self.agents)):
                est_other_goals.append(self.agents[i].estimate_other_goal(other_states[i], other_actions[i]))
           
            #calcurate total reward
            total_reward += sum(rewards)
            self.logger.add_experience(step_n, agents_state, actions, rewards, total_reward, agents_goal, est_other_goals)
           
            #move next_state
            for i in range(len(self.agents)):
                agents_state[i] = next_states[i]
           
            step_n += 1
            
        #calcurate estimate rate and add to logger 
        for i in range(len(self.agents)):
            if other_goals[i] == self.agents[i].get_est_other_goal():
                self.estimate_counts[i] += 1
            self.estimate_rates[i] = self.estimate_counts[i]/(episode_n+1)
            self.logger.add_estimate_rate(i, self.estimate_rates[i])
                
        #save with csv file
        if episode_n % self.report_interval == 0:
            self.logger.state_transition_write_csv(episode_n)
            for i in range(len(self.agents)):
                self.logger.all_q_table_write_csv(episode_n, self.agents[i].Q, i,self.env.row_length, self.env.column_length)
            
        return total_reward, step_n

class SOMQLearningTrainer_Gchange(SOMQLearningTrainer):

    def __init__(self, agents, env, episode=1, report_interval=50, dirname=None, seed=1):
        super().__init__(agents, env, episode, report_interval, dirname, seed)

    def all_seed_train(self):
        for i in range(self.seed):
            print("{} seed train".format(i+1))
            self.reset()
            self.one_seed_train()

        self.logger.show_ave_estimate_rate()
        self.logger.show_ave_estimate_rate_400_ave()

        self.logger.save_all_seed_total_reward_ave_graph()

    def one_seed_train(self):
        #train loop
        self.logger.seed_count()

        for i in range(self.episode):
            if (i+1) % self.report_interval == 0:
                reward, step = self.one_episode(i+1)
                print("Episode {}: Agent gets {} reward. {} step".format(i+1, reward, step))
            else:
                self.one_episode(i+1)

        #estimate transition save
        self.logger.show_estimate_rate()
        self.logger.show_estimate_rate_400_ave()

        #show estimate rate
        """
        for i in range(len(self.agents)):
            self.estimate_rates[i] = self.estimate_counts[i]/self.episode
            print("Agent {} rate: {}".format(i, self.estimate_rates[i]))
        """

        #save the total_reward_graph
        self.logger.save_total_reward_ave_graph()

    def one_episode(self, episode_n):
        #init state & goal
        agents_state=[]
        agents_done=[]
        agents_goal=[]

        agents_state = self.env.reset()
        for i in range(len(self.agents)):
            agents_done.append(False)
            self.agents[i].reset(agents_state[i], self.env.get_goal_states())
            agents_goal.append(self.agents[i].get_my_goal())

        #TODO change general process
        other_goals=[self.agents[1].get_my_goal(), self.agents[0].get_my_goal()]

        #環境側に各エージェントの目的をセット
        self.env.set_agents_goal(agents_goal)

        total_reward = 0

        self.logger.init_list_of_dict()
        step_n = 1

        while False in agents_done:
            actions, next_states, rewards=[],[],[]
            #choice actions
            for i in range(len(self.agents)):
                actions.append(self.agents[i].greedy_act(agents_state[i]))
            
            #one step
            for i in range(len(self.agents)):
                if agents_done[i] == False:
                    next_state, reward, done = self.env.step(i,actions[i])
                else:
                    actions[i]=self.env.actions[4] #STAY
                    next_state, reward, done = self.env.get_finish_state(i), int(0), True

                next_states.append(next_state)
                rewards.append(reward)
                agents_done[i]=done
            
            #is_colision??? 
            #rewards, agents_done = self.env.colision_judge(next_states, rewards, agents_done)

            #TODO generate the process
            #make other_states & action 一旦入れ替えるでオケかな
         
            other_states=[agents_state[1], agents_state[0]]
            other_actions=[actions[1], actions[0]]
            
            
            #learn
            for i in range(len(self.agents)):
                self.agents[i].learn(agents_state[i], next_states[i], actions[i], rewards[i])
           
            est_other_goals=[]
            #estimate
            for i in range(len(self.agents)):
                est_other_goals.append(self.agents[i].estimate_other_goal(other_states[i], other_actions[i]))

            for i in range(len(self.agents)):
                if other_goals[i] == self.agents[i].get_est_other_goal():
                    self.logger.add_is_estimate(i, True)
                else:
                    self.logger.add_is_estimate(i, False)
            
            #calcurate total reward
            total_reward += sum(rewards)

            #log_experience
            self.logger.add_experience(step_n, agents_state, actions, rewards, total_reward, agents_goal, est_other_goals)
            
            #log estimation score
            for i in range(len(self.agents)):
                self.logger.add_estimate_value(i, step_n, self.agents[i].get_estimation_value())

            #update_goal_agent2
            if agents_done[1] == False:
                c_goal = self.agents[1].goal_change(next_states[1])
                if c_goal != agents_goal[1]:
                    agents_goal[1] = c_goal
                    self.env.set_agents_goal(agents_goal)
                    other_goals[0] = c_goal
           
            #move next_state
            for i in range(len(self.agents)):
                agents_state[i] = next_states[i]
           
            step_n += 1

        self.logger.add_total_reward(total_reward)
        self.logger.add_total_reward_ave()
            
        #calcurate estimate rate and add to logger 
        for i in range(len(self.agents)):
            if other_goals[i] == self.agents[i].get_est_other_goal():
                self.estimate_counts[i] += 1
            self.estimate_rates[i] = self.estimate_counts[i]/(episode_n+1)
            self.logger.add_estimate_rate(i, self.estimate_rates[i])

        self.logger.add_estimate_rate_400_ave()
                
        #save with csv file
        if episode_n % self.report_interval == 0:
            self.logger.state_transition_write_csv(episode_n)
            self.logger.estimate_value_write_csv(episode_n)
            for i in range(len(self.agents)):
                self.logger.all_q_table_write_csv(episode_n, self.agents[i].Q, i,self.env.row_length, self.env.column_length)
            
        return total_reward, step_n