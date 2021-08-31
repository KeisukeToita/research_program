import csv
import os
import matplotlib.pyplot as plt
from maze import *

class Logger:
    
    def __init__(self, dirname=None, agent_n=1):
        #make dir
        self.dirname = dirname
        self.agent_n = agent_n
        self.log_dir_mk()

        self.seed = 0
        
        #list of dict
        self.total_reward_log=[]
        self.experience_logs=[]
        for i in range(agent_n):
            agent_exp = []
            self.experience_logs.append(agent_exp)

        #labels
        self.state_transition_header_label=['Step', 'state', 'action', 'reward', 'total_reward']
        self.Q_table_header_label = ['State', 'UP', 'DOWN', 'LEFT', 'RIGHT']
        self.state_transition_with_goal_header_label=['Step', 'state', 'action', 'reward', 'total_reward', 'my_goal', 'est_other_goal']

        #list of graph
        self.all_seed_estimate_rate_logs = []
        self.estimate_rate_logs=[[], []]

    #初期化用
    def init_exp_log(self):
        self.experience_logs=[]
        for i in range(self.agent_n):
            agent_exp = []
            self.experience_logs.append(agent_exp)
        return

    def init_tore_log(self):
        self.total_reward_log=[]
        return

    def log_dir_mk(self):
        #記録用のディレクトリ名の格納
        self.reward_dir = self.dirname+"reward/"
        self.transition_dir = self.dirname+"agents_transition/"
        self.policy_dir = self.dirname+"policy/"

    def seed_count(self):
        self.seed += 1
        self.one_seed_dir_make()
        self.estimate_rate_logs=[[], []]        

    def one_seed_dir_make(self):
        rewdir = self.reward_dir + "seed" + str(self.seed) + "/"
        tradir = self.transition_dir + "seed" + str(self.seed) + "/"
        poldir = self.policy_dir + "seed" + str(self.seed) + "/"

        self.Agents_transition_dir=[]
        self.Agents_reward_dir=[]
        self.Agents_policy_dir=[]
        for i in range(self.agent_n):
            tAgent_dir = tradir+"Agent"+str(i+1)+"/"
            pAgent_dir = poldir+"Agent"+str(i+1)+"/"
            rAgent_dir = rewdir+"Agent"+str(i+1)+"/"
            self.Agents_transition_dir.append(tAgent_dir)
            self.Agents_policy_dir.append(pAgent_dir)
            self.Agents_reward_dir.append(rAgent_dir)
            os.makedirs(tAgent_dir)
            os.makedirs(pAgent_dir)
            os.makedirs(rAgent_dir)

    #状態遷移の記録用関数
    def add_experience(self, step_n, agent_state, a, reward, total_reward, my_goals=None, est_other_goals=None):
        if my_goals == None:
            for i in range(self.agent_n):
                data_dict = {'Step': step_n, 'state': agent_state[i].repr(), 'action': a[i], 'reward': reward[i], 'total_reward': total_reward}
                self.experience_logs[i].append(data_dict)
        else:
            for i in range(self.agent_n):
                data_dict = {'Step': step_n,
                            'state': agent_state[i].repr(),
                            'action': a[i],
                            'reward': reward[i],
                            'total_reward': total_reward,
                            'my_goal': my_goals[i],
                            'est_other_goal': est_other_goals[i]}
                self.experience_logs[i].append(data_dict)

    def add_total_reward(self, total_reward):
        self.total_reward_log.append(total_reward)
        return

    def state_transition_write_csv(self, episode_n):
        is_goal_len = len(self.state_transition_with_goal_header_label)
        not_goal_len = len(self.state_transition_header_label)

        if len(self.experience_logs[0]) == not_goal_len:
            for i in range(self.agent_n):
                filename = self.Agents_transition_dir[i]+"episode"+str(episode_n)+".csv"
                with open(filename, 'w', encoding='shift-jis', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.state_transition_header_label)
                    writer.writeheader()
                    writer.writerows(self.experience_logs[i])

        elif len(self.experience_logs[0]) == is_goal_len:
            for i in range(self.agent_n):
                filename = self.Agents_transition_dir[i]+"episode"+str(episode_n)+".csv"
                with open(filename, 'w', encoding='shift-jis', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.state_transition_with_goal_header_label)
                    writer.writeheader()
                    writer.writerows(self.experience_logs[i])

    #Qテーブル記録用関数
    def q_table_write_csv(self, episode_n, Q, agent_n, row, column, my_goal=None, other_goal=None):
        Q_dict=[]
        for i in range(row):
            for j in range(column):
                Q_dict.append({'State':State(i,j).repr(),
                                'UP':Q[State(i,j).repr()][0],
                                'DOWN':Q[State(i,j).repr()][1],
                                'LEFT':Q[State(i,j).repr()][2],
                                'RIGHT':Q[State(i,j).repr()][3]})

        if my_goal == None:#not goal
            filename = self.Agents_policy_dir[agent_n]+"episode"+str(episode_n)+".csv"
        else:#goal
            filename = self.Agents_policy_dir[agent_n]+"episode"+str(episode_n)+"/Q-table"+str(my_goal)+str(other_goal)+".csv"

        with open(filename, 'w', encoding='shift-jis', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.Q_table_header_label)
                writer.writeheader()
                writer.writerows(Q_dict)

    def all_q_table_write_csv(self, episode_n, Q, agent_n, row, column):
        goal_num = 4
        os.mkdir(self.Agents_policy_dir[agent_n]+"episode"+str(episode_n)+"/")
        for i in range(goal_num):
            for j in range(goal_num):
                self.q_table_write_csv(episode_n, Q[i][j], agent_n, row, column, i, j)

    #estimate_rate_log
    def add_estimate_rate(self, agent_num, rate):
        self.estimate_rate_logs[agent_num].append(rate)

    def show_estimate_rate(self):
        #export as graph
        figs = []
        axs = []
        for i in range(self.agent_n):
            figs.append(plt.figure())
            axs.append(figs[i].add_subplot(1,1,1))
            axs[i].plot(self.estimate_rate_logs[i])
            axs[i].set_xlabel("episode")
            axs[i].set_ylabel("rate")
            axs[i].set_title("Agent {} estimate_rate".format(i+1))
            axs[i].set_ylim(0, 1)
            figs[i].savefig(self.Agents_reward_dir[i]+"estimate_rate.png")
            
        plt.close()

        #export as txt file
        for i in range(self.agent_n):
            f=open(self.Agents_reward_dir[i]+'Agent{}_estimate_rate.txt'.format(i+1), 'x')
            for j, rate in enumerate(self.estimate_rate_logs[i]):
                f.write("Episode{} : {}\n".format(j, rate))
            f.close()

        self.all_seed_estimate_rate_logs.append(copy.deepcopy(self.estimate_rate_logs))
        
        

class Logger_with_goal(Logger):
    def __init__(self, dirname=None, agent_n=1):
        super().__init__(dirname, agent_n)

        self.experience_with_goal_logs = []
        for i in range(agent_n):
            agent_exp = []
            self.experience_with_goal_logs.append(agent_exp)

    def init_exp_log(self):
        super().init_exp_log()
        self.experience_with_goal_logs = []
        for i in range(self.agent_n):
            agent_exp = []
            self.experience_with_goal_logs.append(agent_exp)

    def add_experience_with_goal(self, step_n, agent_state, a, reward, total_reward, my_goals, est_other_goals):
        for i in range(self.agent_n):
            data_dict = {'Step': step_n,
                         'state': agent_state[i].repr(),
                         'action': a[i],
                         'reward': reward[i],
                         'total_reward': total_reward,
                         'my_goal': my_goals[i],
                         'est_other_goal': est_other_goals[i]}

            self.experience_with_goal_logs[i].append(data_dict)

    def state_transition_with_goal_write_csv(self, episode_n):
        for i in range(self.agent_n):
            filename = self.Agents_transition_dir[i]+"episode"+str(episode_n)+".csv"
            with open(filename, 'w', encoding='shift-jis', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.state_transition_with_goal_header_label)
                writer.writeheader()
                writer.writerows(self.experience_with_goal_logs[i])

    def show_ave_estimate_rate(self):
        
        estimate_rate_average = [[], []]
        for i in range(self.agent_n):
            for j in range(len(self.all_seed_estimate_rate_logs[0][0])):
                total_rate = 0
                for k in range(len(self.all_seed_estimate_rate_logs)):
                    total_rate += self.all_seed_estimate_rate_logs[k][i][j]
                ave = total_rate / len(self.all_seed_estimate_rate_logs)
                estimate_rate_average[i].append(ave)

        figs = []
        axs = []
        for i in range(self.agent_n):
            figs.append(plt.figure())
            axs.append(figs[i].add_subplot(1,1,1))
            axs[i].plot(estimate_rate_average[i])
            axs[i].set_xlabel("episode")
            axs[i].set_ylabel("rate")
            axs[i].set_title("Agent {} all estimate_rate".format(i+1))
            axs[i].set_ylim(0, 1)
            figs[i].savefig(self.reward_dir+"agent{}_estimate_rate.png".format(i+1))

        #export as txt file
        for i in range(self.agent_n):
            f=open(self.reward_dir+'Agent{}_estimate_rate_ave.txt'.format(i+1), 'x')
            for j, rate in enumerate(estimate_rate_average[i]):
                f.write("Episode{} : {}\n".format(j, rate))
            f.close()
        