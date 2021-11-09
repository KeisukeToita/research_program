import csv
import os
import matplotlib.pyplot as plt
from maze_8direction import *
from collections import deque, defaultdict
import numpy as np

class AllSeedLogger:
    def __init__(self, dirname, agent_n):
        self.dirname = dirname
        self.agent_n = agent_n
        self.log_dir_mk()

        self.all_seed_estimate_goal_rate_move_ave = []
        self.all_seed_estimate_actplan_rate_move_ave = []
        self.all_seed_total_reward_ave = []
        self.colision_count_list = []

    def log_dir_mk(self):
        self.reward_dir = self.dirname+"reward/"
        self.transition_dir = self.dirname+"agents_transition/"
        self.policy_dir = self.dirname+"policy/"
    
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

    def save_all_estimate_rate_move_ave(self, mode):
        estimate_rate_average = []
        if mode == "GOAL":
            the_logs = np.array(self.all_seed_estimate_goal_rate_move_ave)
            png_fname = "all_agent{}_estimate_goal_rate_move_ave"
        elif mode == "ACTPLAN":
            the_logs = np.array(self.all_seed_estimate_actplan_rate_move_ave)
            png_fname = "all_agent{}_estimate_actplan_rate_move_ave"
        seed_n = len(the_logs)
        min_len = 20000000 #max_len
        for i in range(seed_n):
            min_len = min(min_len, len(the_logs[i, 0]))
    
        #data make
        for i in range(self.agent_n):
            vlist = []
            for k in range(seed_n):
                one_seed_list = []
                for j in range(min_len):
                    one_seed_list.append(the_logs[k][i][j])
                vlist.append(one_seed_list)

            one_agent_list = []
            for j in range(min_len):
                total_rate = 0
                for k in range(seed_n):
                    total_rate += vlist[k][j]
                ave = total_rate / seed_n
                one_agent_list.append(ave)
            estimate_rate_average.append(one_agent_list)

        #export as png file
        for i in range(self.agent_n):
            self.save_graph(
                title=png_fname.format(i+1),
                xlabel="episode",
                ylabel="rate",
                fname=self.reward_dir+png_fname.format(i+1)+".png",
                ylim=(0, 1),
                vlist=estimate_rate_average[i]
            )

        #export as csv file
    
        for i in range(self.agent_n):
            with open(self.reward_dir+png_fname.format(i+1)+".csv", 'w', newline='') as f:
                estimate_rate_average[i] = [[x] for x in estimate_rate_average[i]]
                writer = csv.writer(f)
                writer.writerows(estimate_rate_average[i])

    def add_one_seed_data(self, vlist):
        self.all_seed_data = vlist
    
    def save_all_total_reward_move_ave(self):
        png_fname = "total_reward_500_move_ave"
        total_reward_list = 0
        seed_n = len(self.all_seed_total_reward_ave)
        for i in range(seed_n):
            total_reward_list += np.array(self.all_seed_total_reward_ave[i])
        total_reward_list = np.array(total_reward_list) / seed_n

        #export as png file
        self.save_graph(
            title=png_fname,
            xlabel="episode",
            ylabel="total_reward",
            fname=self.reward_dir+png_fname+".png",
            ylim=(-20, 20),
            vlist=total_reward_list
        )

        #export as csv file
        with open(self.reward_dir+png_fname+".csv", 'w', newline='') as f:
            total_reward_list = [[x] for x in total_reward_list]
            writer = csv.writer(f)
            writer.writerows(total_reward_list)

    def write_colision_count_ave(self):
        colision_count_list = [[x] for x in self.colision_count_list]
        with open(self.reward_dir+"colision_count_ave.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(colision_count_list)
            

    def make_data(self):
        vlist = np.array(self.all_seed_data)
        for i in range(len(vlist)):
            self.all_seed_estimate_goal_rate_move_ave.append(vlist[i][0])
            self.all_seed_estimate_actplan_rate_move_ave.append(vlist[i][1])
            self.all_seed_total_reward_ave.append(vlist[i][2])
            self.colision_count_list.append(vlist[i][3])


class Actplan_Logger:
    
    def __init__(self, dirname=None, agent_n=1):
        #make dir
        self.dirname = dirname
        self.agent_n = agent_n
        self.log_dir_mk()
        
        #list (about reward)
        self.total_reward_log = deque()
        self.total_reward_ave_log = []
        self.all_seed_total_reward_ave_log = []

        #list of dict
        self.estimate_actplan_one_episode = []
        self.experience_logs=[]
        self.estimate_value_logs=[]
        for i in range(agent_n):
            self.experience_logs.append([])
            self.estimate_value_logs.append([])
            self.estimate_actplan_one_episode.append(deque())

        #labels
        self.state_transition_header_label=['Step', 'state', 'action', 'reward', 'total_reward']
        self.Q_table_header_label = ['State', '↑', '↑→', '→', '↓→', '↓', '←↓', '←', '←↑']
        self.state_transition_with_goal_header_label=['Step', 'state', 'action', 'reward', 'total_reward', 'my_goal', 'est_other_goal']
        self.state_transition_with_actplan_relative_state_header_label=['Step',
                                                                        'state',
                                                                        'direction',
                                                                        'other_direction',
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
        self.all_seed_estimate_goal_rate_move_ave = []
        self.all_seed_estimate_actplan_rate_move_ave = []
            
    #about init func
    def init_list_of_dict(self):
        self.estimate_actplan_one_episode = []
        self.experience_logs=[]
        self.estimate_value_logs=[]
        for i in range(self.agent_n):
            self.experience_logs.append([])
            self.estimate_value_logs.append([])
            self.estimate_actplan_one_episode.append(deque())

    def log_dir_mk(self):
        self.reward_dir = self.dirname+"reward/"
        self.transition_dir = self.dirname+"agents_transition/"
        self.policy_dir = self.dirname+"policy/"

    def seed_count(self, seed):
        self.one_seed_dir_make(seed)
        #about reward
        self.total_reward_log=deque()
        self.total_reward_ave_log = []
        #about estimate goal
        self.estimate_goal_queues = []
        self.estimate_goal_rate_ave = []
        for i in range(self.agent_n):
            self.estimate_goal_queues.append(deque())
            self.estimate_goal_rate_ave.append([])
        #about estimate actplan
        self.estimate_actplan_queues = []
        self.estimate_actplan_rate_ave = []
        for i in range(self.agent_n):
            self.estimate_actplan_queues.append(deque())
            self.estimate_actplan_rate_ave.append([]) 

    def one_seed_dir_make(self, seed):
        rewdir = self.reward_dir + "seed" + str(seed) + "/"
        tradir = self.transition_dir + "seed" + str(seed) + "/"
        poldir = self.policy_dir + "seed" + str(seed) + "/"

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
                       my_direction=None,
                       other_direction=None,
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
                                'direction':my_direction[i],
                                'other_direction':other_direction[i],
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
                                'direction':"not_needed",
                                'other_direction':"not_needed",
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
            actplan_str = ["str", "go_r", "go_l", "no"]
            filename = self.Agents_policy_dir[agent_n]+"episode"+str(episode_n)+"/actplan/agent"+str(agent_n+1)+"episode"+str(episode_n)+"actplanQ-table"+str(my_goal)+actplan_str[other_goal]+".csv"

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
    def add_is_estimate_actplan_one_episode(self, agent_num, is_estimate):
        self.estimate_actplan_one_episode[agent_num].append(is_estimate)
        
    def add_is_estimate(self, agent_num, is_estimate=None, mode="ACTPLAN"):
        step_average = 500
        if mode == "GOAL":
            self.estimate_goal_queues[agent_num].append(is_estimate)
            if len(self.estimate_goal_queues[agent_num]) > step_average:
                self.estimate_goal_queues[agent_num].popleft()
        elif mode == "ACTPLAN":#is_estimate is value
            if len(self.estimate_actplan_one_episode[agent_num]) != 0:
                value = self.estimate_actplan_one_episode[agent_num].count(True)/len(self.estimate_actplan_one_episode[agent_num])
                self.estimate_actplan_queues[agent_num].append(value)
            
            if len(self.estimate_actplan_queues[agent_num]) > step_average:
                self.estimate_actplan_queues[agent_num].popleft()

    def add_estimate_rate_move_ave(self, mode):
        if mode == "GOAL":
            for i in range(self.agent_n):
                rate = self.estimate_goal_queues[i].count(True)/len(self.estimate_goal_queues[i])
                self.estimate_goal_rate_ave[i].append(rate)
        elif mode == "ACTPLAN":#is_estimate is value
            for i in range(self.agent_n):
                if len(self.estimate_actplan_queues[i]) != 0:
                    rate = sum(self.estimate_actplan_queues[i])/len(self.estimate_actplan_queues[i])
                    self.estimate_actplan_rate_ave[i].append(rate)
    
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

    def save_estimate_rate_move_ave(self, mode):
        if mode == "GOAL":
            #export as graph
            for i in range(self.agent_n):
                self.save_graph(
                    title="Agent {} estimate_goal_rate_move_ave".format(i+1),
                    xlabel="episode",
                    ylabel="rate",
                    fname=self.Agents_reward_dir[i]+"agent{}_estimate_goal_rate_move_ave.png".format(i+1),
                    ylim=(0, 1),
                    vlist=self.estimate_goal_rate_ave[i]
                )
            #export as txt file
            for i in range(self.agent_n):
                f=open(self.Agents_reward_dir[i]+'Agent{}_estimate_goal_rate_move_ave.txt'.format(i+1), 'x')
                for j, rate in enumerate(self.estimate_goal_rate_ave[i]):
                    f.write("Episode{} : {}\n".format(j, rate))
                f.close()
            self.all_seed_estimate_goal_rate_move_ave.append(copy.deepcopy(self.estimate_goal_rate_ave))
        elif mode == "ACTPLAN":
            #export as graph
            for i in range(self.agent_n):
                self.save_graph(
                    title="Agent {} estimate_actplan_rate_move_ave".format(i+1),
                    xlabel="episode",
                    ylabel="rate",
                    fname=self.Agents_reward_dir[i]+"agent{}_estimate_actplan_rate_move_ave.png".format(i+1),
                    ylim=(0, 1),
                    vlist=self.estimate_actplan_rate_ave[i]
                )
            #export as txt file
            for i in range(self.agent_n):
                f=open(self.Agents_reward_dir[i]+'Agent{}_estimate_rate_move_ave.txt'.format(i+1), 'x')
                for j, rate in enumerate(self.estimate_actplan_rate_ave[i]):
                    f.write("Episode{} : {}\n".format(j, rate))
                f.close()
            self.all_seed_estimate_actplan_rate_move_ave.append(copy.deepcopy(self.estimate_actplan_rate_ave))

    def save_all_estimate_rate_move_ave(self, mode):

        estimate_rate_average = []
        if mode == "GOAL":
            the_logs = np.array(self.all_seed_estimate_goal_rate_move_ave)
            png_fname = "all_agent{}_estimate_goal_rate_move_ave"
        elif mode == "ACTPLAN":
            the_logs = np.array(self.all_seed_estimate_actplan_rate_move_ave)
            png_fname = "all_agent{}_estimate_actplan_rate_move_ave"
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
                title=png_fname.format(i+1),
                xlabel="episode",
                ylabel="rate",
                fname=self.reward_dir+png_fname.format(i+1)+".png",
                ylim=(0, 1),
                vlist=estimate_rate_average[i]
            )

        #export as txt file
        for i in range(self.agent_n):
            f=open(self.reward_dir+png_fname.format(i+1)+".txt", 'x')
            for j, rate in enumerate(estimate_rate_average[i]):
                f.write("Episode{} : {}\n".format(j, rate))
            f.close()

class Data_loader:
    SEED_N = 10

    """
    #TODO 見直しと実際のロードができるか確認．
    """

    def __init__(self, load_dir_path, agent_n):
        self.dir_path = load_dir_path
        self.agent_n = agent_n

        self.reward_dir = self.dir_path+"reward/"
        self.transition_dir = self.dir_path+"agents_transition/"
        self.policy_dir = self.dir_path+"policy/"

        self.Agents_transition_dir=[]
        self.Agents_reward_dir=[]
        self.Agents_policy_dir=[]

        for seed in range(self.SEED_N):
            rewdir = self.reward_dir + "seed" + str(seed+1) + "/"
            tradir = self.transition_dir + "seed" + str(seed+1) + "/"
            poldir = self.policy_dir + "seed" + str(seed+1) + "/"
            
            self.Agents_transition_dir.append(tradir)
            self.Agents_policy_dir.append(poldir)
            self.Agents_reward_dir.append(rewdir)
            
    def goalQ_table_load(self, number):
        goal_n = 4
        seed = 0

        goalQ = []
        for i in range(goal_n):
            goalQ.append([])
        for i in range(goal_n):
            for j in range(goal_n):
                goalQ[i].append(defaultdict(lambda: [0] * 8))

        for i in range(goal_n):
            for j in range(goal_n):
                filename = self.Agents_policy_dir[seed]+"Agent"+str(number+1)+"/episode200000/goal/agent"+str(number+1)+"episode200000goalQ-table"+str(i)+str(j)+".csv"
                
                with open(filename, 'r', encoding='shift-jis', newline='') as f:
                    Qlist = csv.reader(f,  delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
                    header = next(Qlist)

                    for Q_value in Qlist:
                        for k in range(8):
                            goalQ[i][j][Q_value[0]][k] = float(Q_value[k+1])
        
        return goalQ

                
    def actplanQ_table_load(self, number):
        actplan_n = 4
        direction_n = 4
        seed = 0

        actplanQ = []
        for i in range(actplan_n):
            actplanQ.append([])
        for i in range(actplan_n):
            for j in range(direction_n):
                actplanQ[i].append(defaultdict(lambda: [0] * 8))
        
        actplan_str = ["str", "go_r", "go_l", "no"]

        for i in range(actplan_n):
            for j in range(direction_n):
                filename = self.Agents_policy_dir[seed]+"Agent"+str(number+1)+"/episode200000/actplan/agent"+str(number+1)+"episode200000actplanQ-table"+str(i)+actplan_str[j]+".csv"
            #TODO 要見直し!!!! csv.readerの使用を確認しよう
                with open(filename, 'r', encoding='shift-jis', newline='') as f:     
                    Qlist = csv.reader(f,  delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
                    header = next(Qlist)
                        
                    for Q_value in Qlist:
                        for k in range(8):
                            actplanQ[i][j][Q_value[0]][k] = float(Q_value[k+1])

        return actplanQ