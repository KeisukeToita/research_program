import csv
import os

class Logger:
    
    def __init__(self, dirname=None, agent_n=1):
        self.dirname = dirname
        self.agent_n = agent_n
        self.log_dir_mk()
        
        self.total_reward_log=[]
        self.experience_logs=[]
        for i in range(agent_n):
            agent_exp = []
            self.experience_logs.append(agent_exp)

        self.file_name=None
        self.state_transition_header_label=['Step', 'state', 'action', 'reward', 'total_reward']

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
        #記録用のディレクトリを作成
        self.reward_dir = self.dirname+"reward/"
        os.mkdir(self.reward_dir)
        self.transition_dir = self.dirname+"agents_transition/"
        os.mkdir(self.transition_dir)
        self.Agents_transition_dir=[]
        for i in range(self.agent_n):
            Agent_dir = self.transition_dir+"Agent"+str(i+1)+"/"
            self.Agents_transition_dir.append(Agent_dir)
            os.mkdir(Agent_dir)

        self.polisy_dir = self.dirname+"policy/"
        os.mkdir(self.polisy_dir)


    #状態遷移の記録用関数
    def add_experience(self, step_n, agent_state, a, reward, total_reward):
        for i in range(self.agent_n):
            data_dict = {'Step': step_n, 'state': agent_state[i].repr(), 'action': a[i], 'reward': reward[i], 'total_reward': total_reward}
            self.experience_logs[i].append(data_dict)
        return

    def add_total_reward(self, total_reward):
        self.total_reward_log.append(total_reward)
        return

    def state_transition_write_csv(self, episode_n):

        for i in range(self.agent_n):
            filename = self.Agents_transition_dir[i]+"episode"+str(episode_n)+".csv"
            with open(filename, 'w', encoding='shift-jis', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.state_transition_header_label)
                writer.writeheader()
                writer.writerows(self.experience_logs[i])

    #値のセット
    def set_file_name(self, file_name):
        self.file_name = file_name
        return