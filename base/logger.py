import csv

class Logger:
    
    def __init__(self, dirname=None):
        self.dirname = dirname

        self.total_reward_log=[]
        self.experience_log=[]

        self.file_name=None
        self.state_transition_header_label=['Step', 'state', 'action', 'reward', 'total_reward']

    #初期化用
    def init_exp_log(self):
        self.experience_log=[]
        return

    def init_tore_log(self):
        self.total_reward_log=[]
        return

    #状態遷移の記録用関数
    def add_experience(self, step_n, agent_state, a, reward, total_reward):
        data_dict = {'Step': step_n, 'state': agent_state.repr(), 'action': a, 'reward': reward, 'total_reward': total_reward}
        self.experience_log.append(data_dict)
        return

    def add_total_reward(self, total_reward):
        self.total_reward_log.append(total_reward)
        return

    def state_transition_write_csv(self, episode_n):

        filename = self.dirname+"episode"+str(episode_n)+".csv"

        with open(filename, 'w', encoding='shift-jis', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.state_transition_header_label)
            writer.writeheader()
            writer.writerows(self.experience_log)

    #値のセット
    def set_file_name(self, file_name):
        self.file_name = file_name
        return