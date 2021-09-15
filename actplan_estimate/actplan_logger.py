import csv
import os
import matplotlib.pyplot as plt
from maze_8direction import *
from collections import deque
import numpy as np

class Actplan_Logger:
    
    def __init__(self, dirname=None, agent_n=1):
        #make dir
        self.dirname = dirname
        self.agent_n = agent_n
        self.log_dir_mk()

        self.seed = 0
        
        #list (about reward)
        self.total_reward_log = deque()
        self.total_reward_ave_log = []
        self.all_seed_total_reward_ave_log = []

        #list of dict
        self.experience_logs=[]
        self.estimate_value_logs=[]
        for i in range(agent_n):
            agent_exp = []
            est = []
            self.experience_logs.append(agent_exp)
            self.estimate_value_logs.append(est)

        #labels
        self.state_transition_header_label=['Step', 'state', 'action', 'reward', 'total_reward']
        self.Q_table_header_label = ['State', '↑', '↑→', '→', '↓→', '↓', '←↓', '←', '←↑']
        self.state_transition_with_goal_header_label=['Step', 'state', 'action', 'reward', 'total_reward', 'my_goal', 'est_other_goal']
        self.state_transition_with_actplan_relative_state_header_label=['Step',
                                                                        'state',
                                                                        'other_state',
                                                                        'relative_state',
                                                                        'mode',
                                                                        'action',
                                                                        'reward',
                                                                        'total_reward',
                                                                        'my_goal',
                                                                        'est_other_goal',
                                                                        'my_actplan',
                                                                        'est_other_actplan']
        self.goal_estimate_label = ['step', 'goal1', 'goal2','goal3','goal4']
        self.actplan_estimate_label = ['step', 'straight', 'go_left', 'go_right']

        #list of graph
        self.all_seed_estimate_rate_logs = []
        self.all_seed_estimate_rate_400_ave = []

        self.estimate_rate_logs=[]
        self.estimate_queues = []
        self.estimate_rate_400_ave = []
        for i in range(agent_n):
            self.estimate_rate_logs.append([])
            self.estimate_queues.append(deque())
            self.estimate_rate_400_ave.append([])
            

    #初期化用
    def init_list_of_dict(self):
        self.experience_logs=[]
        self.estimate_value_logs=[]
        for i in range(self.agent_n):
            agent_exp = []
            est = []
            self.experience_logs.append(agent_exp)
            self.estimate_value_logs.append(est)

    def log_dir_mk(self):
        self.reward_dir = self.dirname+"reward/"
        self.transition_dir = self.dirname+"agents_transition/"
        self.policy_dir = self.dirname+"policy/"

    def seed_count(self):
        self.seed += 1
        self.one_seed_dir_make()

        self.total_reward_log=deque()
        self.total_reward_ave_log = []

        self.estimate_rate_logs=[]
        self.estimate_queues = []
        self.estimate_rate_400_ave = []
        for i in range(self.agent_n):
            self.estimate_rate_logs.append([])
            self.estimate_queues.append(deque())
            self.estimate_rate_400_ave.append([]) 

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

    def add_total_reward(self, total_reward):
        episode_ave = 500
        self.total_reward_log.append(total_reward)
        if len(self.total_reward_log) > episode_ave:
            self.total_reward_log.popleft()

    def add_total_reward_ave(self):
        ave = sum(self.total_reward_log)/len(self.total_reward_log)
        self.total_reward_ave_log.append(ave)
        

    #state_transition_log func
    def add_experience(self,
                       step_n,
                       agent_state,
                       a,
                       reward,
                       total_reward,
                       label,
                       mode,
                       other_states=None,
                       relative_states=None,
                       my_goals=None,
                       est_other_goals=None,
                       my_actplans=None,
                       est_other_actplans=None):
        for i in range(self.agent_n):
            if label == "NONE":
                data_dict = {'Step': step_n,
                            'state': agent_state[i].repr(),
                            'action': a[i],
                            'reward': reward[i],
                            'total_reward': total_reward}
            elif label == "with GOAL":
                data_dict = {'Step': step_n,
                            'state': agent_state[i].repr(),
                            'action': a[i],
                            'reward': reward[i],
                            'total_reward': total_reward,
                            'my_goal': my_goals[i],
                            'est_other_goal': est_other_goals[i]}
            elif label == "with ACTPLAN":
                if mode == "ACTPLAN":
                    data_dict = {'Step':step_n,
                                'state':agent_state[i].repr(),
                                "other_state":other_states[i].repr(),
                                'relative_state':relative_states[i].repr(),
                                'mode':mode,
                                'action':a[i],
                                'reward':reward[i],
                                'total_reward':total_reward,
                                'my_goal':my_goals[i],
                                'est_other_goal':est_other_goals[i],
                                'my_actplan':my_actplans[i],
                                'est_other_actplan':est_other_actplans[i]}
                elif mode == "GOAL":
                    data_dict = {'Step':step_n,
                                'state':agent_state[i].repr(),
                                "other_state":"not_needed",
                                'relative_state':"not_needed",
                                'mode':mode,
                                'action':a[i],
                                'reward':reward[i],
                                'total_reward':total_reward,
                                'my_goal':my_goals[i],
                                'est_other_goal':est_other_goals[i],
                                'my_actplan':"not_needed",
                                'est_other_actplan':"not_needed"}

            self.experience_logs[i].append(data_dict)

    #state write csv
    def state_transition_write_csv(self, episode_n):
        is_goal_len = len(self.state_transition_with_goal_header_label)
        not_goal_len = len(self.state_transition_header_label)
        with_actplan_len = len(self.state_transition_with_actplan_relative_state_header_label)

        if len(self.experience_logs[0][0]) == not_goal_len:
            for i in range(self.agent_n):
                filename = self.Agents_transition_dir[i]+"agent"+str(i+1)+"episode"+str(episode_n)+"state_transition.csv"
                with open(filename, 'w', encoding='shift-jis', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.state_transition_header_label)
                    writer.writeheader()
                    writer.writerows(self.experience_logs[i])
        elif len(self.experience_logs[0][0]) == is_goal_len:
            for i in range(self.agent_n):
                filename = self.Agents_transition_dir[i]+"agent"+str(i+1)+"episode"+str(episode_n)+"state_transition.csv"
                with open(filename, 'w', encoding='shift-jis', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.state_transition_with_goal_header_label)
                    writer.writeheader()
                    writer.writerows(self.experience_logs[i])
        elif len(self.experience_logs[0][0]) == with_actplan_len:
            for i in range(self.agent_n):
                filename = self.Agents_transition_dir[i]+"agent"+str(i+1)+"episode"+str(episode_n)+"state_transition.csv"
                with open(filename, 'w', encoding='shift-jis', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.state_transition_with_actplan_relative_state_header_label)
                    writer.writeheader()
                    writer.writerows(self.experience_logs[i])

    #Q table save function
    def q_table_write_csv(self, episode_n, Q, agent_n, row, column, mode, my_goal=None, other_goal=None):
        Q_dict=[]
        if mode == "ACTPLAN":
            for i in range(-2, 3):
                for j in range(-2, 3):
                    Q_dict.append({'State':State(i,j).repr(),
                                    '↑':Q[State(i,j).repr()][0],
                                    '↑→':Q[State(i,j).repr()][1],
                                    '→':Q[State(i,j).repr()][2],
                                    '↓→':Q[State(i,j).repr()][3],
                                    '↓':Q[State(i,j).repr()][4],
                                    '←↓':Q[State(i,j).repr()][5],
                                    '←':Q[State(i,j).repr()][6],
                                    '←↑':Q[State(i,j).repr()][7]})
        else:
            for i in range(row):
                for j in range(column):
                    Q_dict.append({'State':State(i,j).repr(),
                                    '↑':Q[State(i,j).repr()][0],
                                    '↑→':Q[State(i,j).repr()][1],
                                    '→':Q[State(i,j).repr()][2],
                                    '↓→':Q[State(i,j).repr()][3],
                                    '↓':Q[State(i,j).repr()][4],
                                    '←↓':Q[State(i,j).repr()][5],
                                    '←':Q[State(i,j).repr()][6],
                                    '←↑':Q[State(i,j).repr()][7]})

        if mode == "NONE":
            filename = self.Agents_policy_dir[agent_n]+"episode"+str(episode_n)+".csv"
        elif mode == "GOAL":#goal
            filename = self.Agents_policy_dir[agent_n]+"episode"+str(episode_n)+"/goal/agent"+str(agent_n+1)+"episode"+str(episode_n)+"goalQ-table"+str(my_goal)+str(other_goal)+".csv"
        elif mode == "ACTPLAN":
            filename = self.Agents_policy_dir[agent_n]+"episode"+str(episode_n)+"/actplan/agent"+str(agent_n+1)+"episode"+str(episode_n)+"actplanQ-table"+str(my_goal)+str(other_goal)+".csv"

        with open(filename, 'w', encoding='shift-jis', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.Q_table_header_label)
                writer.writeheader()
                writer.writerows(Q_dict)

    def all_q_table_write_csv(self, episode_n, Q, agent_n, row, column, mode):
        if mode == "GOAL":
            goal_num = 4
            os.makedirs(self.Agents_policy_dir[agent_n]+"episode"+str(episode_n)+"/goal/")
            for i in range(goal_num):
                for j in range(goal_num):
                    self.q_table_write_csv(episode_n, Q[i][j], agent_n, row, column, mode, i, j)
        elif mode == "ACTPLAN":
            actplan_num = 4
            os.makedirs(self.Agents_policy_dir[agent_n]+"episode"+str(episode_n)+"/actplan/")
            for i in range(actplan_num):
                for j in range(actplan_num):
                    self.q_table_write_csv(episode_n, Q[i][j], agent_n, row, column, mode, i, j)
        

    #estimate_Rate_log
    def add_estimate_rate(self, agent_num, rate):
        self.estimate_rate_logs[agent_num].append(rate)

    def add_is_estimate(self, agent_num, is_estimate):
        step_average = 400
        self.estimate_queues[agent_num].append(is_estimate)
        if len(self.estimate_queues[agent_num]) > step_average:
            self.estimate_queues[agent_num].popleft()
    
    def add_estimate_rate_400_ave(self):
        for i in range(self.agent_n):
            rate = self.estimate_queues[i].count(True)/len(self.estimate_queues[i])
            self.estimate_rate_400_ave[i].append(rate)

    
    #estimate_Value_log & write
    def add_estimate_value(self, agent_n, step_n, values):
        vdict = {'step':step_n, 'goal1':values[0], 'goal2':values[1],'goal3':values[2],'goal4':values[3]}
        self.estimate_value_logs[agent_n].append(vdict)

    def estimate_value_write_csv(self, episode_n):
        for i in range(self.agent_n):
            filename = self.Agents_reward_dir[i]+"episode"+str(episode_n)+".csv"
            with open(filename, 'w', encoding='shift-jis', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.goal_estimate_label)
                writer.writeheader()
                writer.writerows(self.estimate_value_logs[i])

    #save as graph func
    def save_graph(self, title, xlabel, ylabel, fname, ylim, vlist):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(vlist)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(ylim[0], ylim[1])
        fig.savefig(fname)
        plt.close()
        
    def save_total_reward_ave_graph(self):
        self.save_graph(
            title="total_reward_average_500epi",
            xlabel="episode",
            ylabel="reward_ave",
            fname=self.reward_dir+"total_reward_average_500epi_seed{}.png".format(self.seed),
            ylim=(-20, 20),
            vlist=self.total_reward_ave_log
        )

        f=open(self.reward_dir+"total_reward_average_500epi_seed{}.txt".format(self.seed), 'x')
        for i, rate in enumerate(self.total_reward_ave_log):
            f.write("Episode{} : {}\n".format(i, rate))
        f.close()

        self.all_seed_total_reward_ave_log.append(self.total_reward_ave_log)

    def save_all_seed_total_reward_ave_graph(self):
        seed_average = []
        the_logs = np.array(self.all_seed_total_reward_ave_log)
        seed_n = len(the_logs)

        total = 0
        for i in range(seed_n):
            total += the_logs[i]
        seed_average = total / seed_n

        self.save_graph(
            title="total_reward_average_500epi",
            xlabel="episode",
            ylabel="reward_ave",
            fname=self.reward_dir+"all_seed_total_reward_average.png",
            ylim=(-20, 20),
            vlist=seed_average
        )

        f=open(self.reward_dir+"all_seed_total_reward_average.txt", 'x')
        for i, rate in enumerate(self.total_reward_ave_log):
            f.write("Episode{} : {}\n".format(i, rate))
        f.close()


    def show_estimate_rate(self):
        #export as graph  
        for i in range(self.agent_n):
            self.save_graph(
                title="Agent {} all estimate_rate".format(i+1),
                xlabel="episode",
                ylabel="rate",
                fname=self.Agents_reward_dir[i]+"agent{}_estimate_rate.png".format(i+1),
                ylim=(0, 1),
                vlist=self.estimate_rate_logs[i]
            )

        #export as txt file
        for i in range(self.agent_n):
            f=open(self.Agents_reward_dir[i]+'Agent{}_estimate_rate.txt'.format(i+1), 'x')
            for j, rate in enumerate(self.estimate_rate_logs[i]):
                f.write("Episode{} : {}\n".format(j, rate))
            f.close()

        self.all_seed_estimate_rate_logs.append(copy.deepcopy(self.estimate_rate_logs))

    def show_estimate_rate_400_ave(self):
        #export as graph
        for i in range(self.agent_n):
            self.save_graph(
                title="Agent {} estimate_rate_400_ave".format(i+1),
                xlabel="episode",
                ylabel="rate",
                fname=self.Agents_reward_dir[i]+"agent{}_estimate_rate_400_ave.png".format(i+1),
                ylim=(0, 1),
                vlist=self.estimate_rate_400_ave[i]
            )

        #export as txt file
        for i in range(self.agent_n):
            f=open(self.Agents_reward_dir[i]+'Agent{}_estimate_rate_400_ave.txt'.format(i+1), 'x')
            for j, rate in enumerate(self.estimate_rate_logs[i]):
                f.write("Episode{} : {}\n".format(j, rate))
            f.close()

        self.all_seed_estimate_rate_400_ave.append(copy.deepcopy(self.estimate_rate_400_ave))

    def show_ave_estimate_rate(self):
        estimate_rate_average = []
        the_logs = np.array(self.all_seed_estimate_rate_logs)
        seed_n = len(the_logs)

        for i in range(self.agent_n):
            vlist = the_logs[:, i, :]
            total_rate = 0
            for j in range(seed_n):
                total_rate += vlist[j]
            ave = total_rate / seed_n
            estimate_rate_average.append(ave)

        for i in range(self.agent_n):
            self.save_graph(
                title="Agent {} all_estimate_rate".format(i+1),
                xlabel="episode",
                ylabel="rate",
                fname=self.reward_dir+"agent{}_estimate_rate.png".format(i+1),
                ylim=(0, 1),
                vlist=estimate_rate_average[i]
            )

        #export as txt file
        for i in range(self.agent_n):
            f=open(self.reward_dir+'Agent{}_estimate_rate_ave.txt'.format(i+1), 'x')
            for j, rate in enumerate(estimate_rate_average[i]):
                f.write("Episode{} : {}\n".format(j, rate))
            f.close()

    def show_ave_estimate_rate_400_ave(self):

        estimate_rate_average = []
        the_logs = np.array(self.all_seed_estimate_rate_400_ave)
        seed_n = len(the_logs)

        for i in range(self.agent_n):
            vlist = the_logs[:, i, :]
            total_rate = 0
            for j in range(seed_n):
                total_rate += vlist[j]
            ave = total_rate / seed_n
            estimate_rate_average.append(ave)

        for i in range(self.agent_n):
            self.save_graph(
                title="Agent {} all_estimate_rate_400_ave".format(i+1),
                xlabel="episode",
                ylabel="rate",
                fname=self.reward_dir+"agent{}_estimate_rate_400_ave.png".format(i+1),
                ylim=(0, 1),
                vlist=estimate_rate_average[i]
            )

        #export as txt file
        for i in range(self.agent_n):
            f=open(self.reward_dir+'Agent{}_estimate_rate_ave_400_ave.txt'.format(i+1), 'x')
            for j, rate in enumerate(estimate_rate_average[i]):
                f.write("Episode{} : {}\n".format(j, rate))
            f.close()