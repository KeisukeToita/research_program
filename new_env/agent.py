import numpy as np
import copy
from collections import defaultdict
import pprint

class Multi_Agent:
    
    def __init__(self, agent, agent_conf):
        self.n = agent_conf.agent_num
        self.agents = []
        for i in range(self.n):
            agent = copy.deepcopy(agent)
            self.agents.append(agent)
        self.dones = np.zeros(self.n)
        #状態加工クラス
        self.transformer = Transformer()
        self.value_type = agent_conf.value_type

        for i in range(self.n):
            self.agents[i].set_number(i)
        
    def episode_reset(self, agents_goal):
        self.dones = np.zeros(self.n)
        for agent in self.agents:
            agent.episode_reset()
        #about goal
        self.agents_goal = agents_goal
        self.estimate_goal_list = [[np.random.randint(4) for i in range(self.n-1)] for j in range(self.n)]
            
    def act(self, states):
        #状態を連続値から離散値へ変換
        if self.value_type == "DISCRETE":
            states = self.transformer.trans_obs_seq_to_int(states)
        o_s = states.get_other_states()
        actions = []
        for i, agent in enumerate(self.agents):
            s = states.access(i)
            a = agent.act(s, o_s[i], self.agents_goal[i], self.estimate_goal_list[i])
            actions.append(a)
        return actions
    
    def learn(self, states, actions, rewards, next_states):
        #状態を連続値から離散値へ変換
        if self.value_type == "DISCRETE":
            states = self.transformer.trans_obs_seq_to_int(states)
            next_states = self.transformer.trans_obs_seq_to_int(next_states)
        #他エージェントの状態をまとめたリストの取得
        o_s_l = states.get_other_states()
        n_o_s_l = next_states.get_other_states()
        
        for i in range(self.n):
            if not self.dones[i]:
                s = states.access(i)
                n_s = next_states.access(i)
                o_s = o_s_l[i]
                n_o_s = n_o_s_l[i]
                a = actions[i]
                r = rewards[i]
                self.agents[i].learn(s, a, r, n_s, o_s, n_o_s, self.agents_goal[i], self.estimate_goal_list[i])
    
    def estimate_goal(self, m_a_s, o_a_l):
        if self.value_type == "DISCRETE":
            m_a_s = self.transformer.trans_obs_seq_to_int(m_a_s)
        o_s_l = m_a_s.get_other_states()
        for i in range(self.n):
            est_goal_list = self.agents[i].estimate_goal(o_s_l[i], 
                                                         o_a_l[i],
                                                         self.agents_goal[i], 
                                                         self.estimate_goal_list[i])
            self.estimate_goal_list[i] = est_goal_list
            
    def update_dones(self, dones):
        self.dones = copy.deepcopy(dones)
        
    def set_epsilon(self, epsilon):
        for i in range(self.n):
            self.agents[i].epsilon = epsilon
    
    def check_estimate(self):
        """
        よくない関数：即席で制作
        """
        c_list = []
        c_list.append(self.agents_goal[0] == self.estimate_goal_list[1][0])
        c_list.append(self.agents_goal[1] == self.estimate_goal_list[0][0])
        
        return c_list
        
class  QLearningAgent:
    def __init__(self, agent_conf, action_space, goal_num):
        self.number = 0
        self.agent_conf = agent_conf
        #モード(ゴールを目指すか，衝突回避をするか)
        self.mode = agent_conf.mode
        #各パラメータ
        self.alpha = agent_conf.alpha
        self.gamma = agent_conf.gamma
        self.epsilon = agent_conf.epsilon
        #行動選択法
        self.act_choice = agent_conf.action_choice
        #行動空間
        self.action_space = action_space
        #各種，状態行動価値関数
        #GOAL_Q
        """
        #TODO 人数に応じた階層構造をつくる
        """
        self.goal_num = goal_num
        self.goalQ = []
        for i in range(self.goal_num):
            self.goalQ.append([])
        for i in range(self.goal_num):
            for j in range(self.goal_num):
                self.goalQ[i].append(self.make_Q_table(1, [0]*self.action_space.n))
        
        #ACTPLAN_Q
        self.actplanQ = self.make_Q_table(agent_conf.agent_num, [0]*self.action_space.n)
        
        #estimate goal
        self.goal_num = goal_num
        self.goal_infer_steps = 0
        self.goal_estimation_value = np.zeros((self.agent_conf.agent_num,goal_num))
        self.goal_estimation_score = np.zeros((self.agent_conf.agent_num,goal_num))
        
    def set_number(self, number):
        self.number = number
    
    def episode_reset(self):
        #estimate goal
        self.goal_infer_steps = 0
        self.goal_estimation_value = np.zeros(self.goal_num)
        self.goal_estimation_score = np.zeros(self.goal_num)
        
    #行動選択
    def act(self, s, o_s=None, g=None, e_g_l=None):
        policy = self.policy_access(s, o_s, g, e_g_l)
        if self.act_choice == "EPSILON":
            return self.e_greedy_act(policy)
        elif self.act_choice == "SOFTMAX":
            return self.softmax_act(policy)
        
    def e_greedy_act(self, policy):
        if np.random.random() < self.epsilon:
            # random
            return np.random.randint(self.action_space.n)
        else:
            # choice best action
            if sum(policy) != 0:
                return np.argmax(policy)
            else:
                return np.random.randint(self.action_space.n)
    def softmax_act(self, policy):
        pass
        
    #learn part
    def learn(self,s,a,r,n_s,o_s,n_o_s, g, e_g_l):
        Q = self.policy_access(s, o_s, goal=g, e_goal_l=e_g_l)
        next_Q = self.policy_access(n_s, n_o_s, goal=g, e_goal_l=e_g_l)
        
        gain = r + self.gamma * max(next_Q)
        estimated = Q[a]
        Q[a] += self.alpha * (gain - estimated)
    
    #estimate part
    def estimate_goal(self, o_s_l, o_a_l, goal, e_g_l):
        self.goal_infer_steps += 1
        estimate_goal_list = []
        
        goal_list = self.my_goal_list(goal, e_g_l)

        for i in range(self.agent_conf.agent_num-1):
            estimate_goal = self.estimate_goal_single(i, o_s_l[i], o_a_l[i], goal_list[i])
            estimate_goal_list.append(estimate_goal)
        
        return estimate_goal_list
            
    def estimate_goal_single(self, agent_num, o_s, o_a, e_g_l):
        policy_each_goal = []
        for other_goal in range(self.goal_num):
            the_policy = self.policy_access(o_s, goal=other_goal, e_goal_l=e_g_l)
            if sum(the_policy)!=0:
                policy_each_goal.append(the_policy[o_a]/ sum(the_policy))
            else:
                policy_each_goal.append(0.01/self.goal_num)
        
        #３人以上
        # self.goal_estimation_value[agent_num] += policy_each_goal
        # self.goal_estimation_score[agent_num] = self.goal_estimation_value[agent_num] / self.goal_infer_steps
        
        self.goal_estimation_value += policy_each_goal
        self.goal_estimation_score = self.goal_estimation_value / self.goal_infer_steps
        
        #update est_other_goal
        # a = np.where(self.goal_estimation_value[agent_num] == max(self.goal_estimation_value[agent_num]))[0]
        a = np.where(self.goal_estimation_value == max(self.goal_estimation_value))[0]
        est_other_goal = np.random.choice(a)
        return est_other_goal
        
    def my_goal_list(self, goal, e_g_l):
        my_goal_list = []
        est_goal_list = copy.deepcopy(e_g_l)
        for i in range(self.agent_conf.agent_num):
            if i == self.number:
                my_goal_list.append(goal)
            else:
                my_goal_list.append(est_goal_list.pop())
        
        est_goal_list = []
        for j in range(self.agent_conf.agent_num):
            if j != self.number:
                o_g_l = [x for i, x in enumerate(my_goal_list) if i != j]
                est_goal_list.append(o_g_l)
        return est_goal_list
        
    #operate Q_table
    def make_Q_table(self, agent_num, v_type):
        if agent_num == 1:
            return copy.deepcopy(defaultdict(lambda:copy.deepcopy(v_type)))
        else:
            return defaultdict(lambda: self.make_Q_table(agent_num-1, v_type))
        
    def print_Q_value(self):
        for key in self.actplanQ.keys():
            pprint.pprint(self.actplanQ[key])
        
    def policy_access(self, my_state, other_states=None, goal=None, e_goal_l=None, actplan=None):
        if self.mode == "GOAL":
            # the_policy = self.goalQ[goal][my_state._repr()]
            the_policy = self.goalQ[goal]
            for e in e_goal_l:
                the_policy = the_policy[e]
            return the_policy[my_state._repr()]    
        elif self.mode == "ACTPLAN":
            the_policy = self.actplanQ[my_state._repr()]
            for o_s in other_states:
                the_policy = the_policy[o_s._repr()]  
        return the_policy
        

class Transformer:
    def __init__(self):
        pass
    
    def trans_obs_seq_to_int(self, observation):
        """
        別ファイルのMulti_State用の関数
        連続値から整数値に変換するよ．
        """
        obs = observation
        for i in range(obs.agent_num):
            s = obs.access(i)
            x, y = s.x, s.y
            trans_s = s.clone()
            trans_s.x, trans_s.y = int(x), int(y)
            obs.set_state(i, trans_s)
        return obs