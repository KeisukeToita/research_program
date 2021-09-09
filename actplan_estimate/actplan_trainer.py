from matplotlib.pyplot import step
from actplan_agent import *
from maze_8direction import *
from actplan_logger import *
import numpy as np
import copy

np.random.seed(0)

class Base_Trainer:
    def __init__(self, agents, env, episode=1, report_interval=50, dirname=None, seed=1, is_colision=0):
        print("~~~ Learning_START ~~~")
        self.env = env
        self.agents = agents

        self.seed = seed
        self.episode = episode
        self.report_interval = report_interval

        self.is_colision = is_colision

        self.logger = Actplan_Logger(dirname, len(agents))

    def seed_reset(self):
        for i in range(len(self.agents)):
            self.agents[i].seed_reset()

    def all_seed_train(self):
        for i in range(self.seed):
            print("{} seed train".format(i+1))
            self.seed_reset()
            self.one_seed_train()        

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
        """
        self.logger.show_estimate_rate()
        """

        #show estimate rate
        """
        for i in range(len(self.agents)):
            self.estimate_rates[i] = self.estimate_counts[i]/self.episode
            print("Agent {} rate: {}".format(i, self.estimate_rates[i]))
        """

    def one_episode(self, episode_n):
        #init state & goal
        agents_state=[]
        agents_done=[]

        agents_state = self.env.reset()
        for i in range(len(self.agents)):
            agents_done.append(False)
            self.agents[i].episode_reset()

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
                    actions[i]=self.env.actions[8] #STAY
                    next_state, reward, done = self.env.get_finish_states(i), int(0), True

                next_states.append(next_state)
                rewards.append(reward)
                agents_done[i]=done
            
            #colision_judge part
            if self.is_colision == 1:
                rewards, agents_done = self.env.colision_judge(next_states, rewards, agents_done)

            #learn
            for i in range(len(self.agents)):
                self.agents[i].learn(agents_state[i], next_states[i], actions[i], rewards[i])
           
            #calcurate total reward
            total_reward += sum(rewards)
            self.logger.add_experience(step_n, agents_state, actions, rewards, total_reward)
           
            #move next_state
            for i in range(len(self.agents)):
                agents_state[i] = next_states[i]
           
            step_n += 1
                
        #save with csv file
        if episode_n % self.report_interval == 0:
            self.logger.state_transition_write_csv(episode_n)
            for i in range(len(self.agents)):
                self.logger.q_table_write_csv(episode_n, self.agents[i].Q, i, self.env.row_length, self.env.column_length)
            
        return total_reward, step_n