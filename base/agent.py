import random
from enum import Enum
import numpy as np
import math
from base_utils import *
from collections import defaultdict

# Agent_Base

class Agent():
    def __init__(self, env):
        self.actions = [Action.UP,Action.DOWN,Action.LEFT,Action.RIGHT]

    def act(self, state):  # TODO 行動確率を返したい．行動を返す関数 & Plannerに行動選択も委託しよう
        a = random.choice(self.actions)
        return a

    def learn(self):  # TODO 学習方法について(引数には必要な要素) & Plannerに役割を分散できるか．
        pass


class MonteCarloAgent(Agent):

    def __init__(self, env, epsilon=0.1, gamma=0.9):
        super().__init__(env)
        self.epsilon = epsilon
        self.reward_log = []
        self.experience_log = []
        self.Q = defaultdict(lambda: [0] * len(self.actions))
        self.N = defaultdict(lambda: [0] * len(self.actions))
        self.gamma = gamma

    def act(self, now_state):  # epsilon-greedy
        if np.random.random() < self.epsilon:
            # ランダム行動選択
            return trans_ntoa(np.random.randint(len(self.actions)))
        else:
            # 現時点の良い行動を選択
            if now_state in self.Q and sum(self.Q[now_state]) != 0:
                return trans_ntoa(np.argmax(self.Q[now_state]))
            else:
                return trans_ntoa(np.random.randint(len(self.actions)))
        

    def init_log(self):  # experienceの初期化
        self.experience_log = []

    def experience_add(self, now_state, action, reward):  # listにどんどんappendしていく
        #現在の状態からある行動によってどれだけの行動を得られたか
        self.experience_log.append({"state": now_state, "action": trans_aton(action), "reward": reward})
    
    def reward_add(self, reward):
        #reward_logに1エピソード分の報酬を追加
        self.reward_log.append(reward)

    def learn(self): #1エピソード終了後に発動
        for i, x in enumerate(self.experience_log):
            s, a = x["state"], x["action"]

            # Calculate discounted future reward of s.
            G, t = 0, 0
            for j in range(i, len(self.experience_log)):
                G += math.pow(self.gamma, t) * self.experience_log[j]["reward"]
                t += 1

            self.N[s][a] += 1  # count of s, a pair
            alpha = 1 / self.N[s][a]
            self.Q[s][a] += alpha * (G - self.Q[s][a])

class QLearningAgent(Agent):
    
    def __init__(self, env, epsilon=0.1, gamma=0.9, alpha=0.1):
        super().__init__(env)
        self.epsilon = epsilon
        self.reward_log = []
        self.Q = defaultdict(lambda: [0] * len(self.actions))
        self.gamma = gamma
        self.alpha = alpha

    def act(self, now_state):  # epsilon-greedy
        if np.random.random() < self.epsilon:
            # ランダム行動選択
            return trans_ntoa(np.random.randint(len(self.actions)))
        else:
            # 現時点の良い行動を選択
            if now_state.repr() in self.Q and sum(self.Q[now_state.repr()]) != 0:
                return trans_ntoa(np.argmax(self.Q[now_state.repr()]))
            else:
                return trans_ntoa(np.random.randint(len(self.actions)))

    def reward_add(self, reward):
        #reward_logに1エピソード分の報酬を追加
        self.reward_log.append(reward)

    def learn(self, s, n_s, action, reward):
        a = trans_aton(action)
        gain = reward + self.gamma * max(self.Q[n_s.repr()])
        estimated = self.Q[s.repr()][a]
        self.Q[s.repr()][a] += self.alpha * (gain - estimated)

class SOMQLearningAgent(Agent):

    def __init__(self, env, number_of_goals, epsilon=0.1, gamma=0.9, alpha=0.1):
        super().__init__(env)
        
        self.number_of_goals = number_of_goals
        self.reward_log = []

        self.Q = []
        for i in range(number_of_goals):
            self.Q.append([])
        for i in range(number_of_goals):
            for j in range(number_of_goals):
                self.Q[i].append(defaultdict(lambda: [0] * len(self.actions)))

        #パラメータ
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        self.infer_steps = 0
        self.estimation_value_other_goal = np.zeros(number_of_goals)
        self.estimation_score = np.zeros(number_of_goals)

        self.my_goal = np.random.randint(number_of_goals)
        self.est_other_goal = np.random.randint(number_of_goals)

        self.policy = self.Q[self.my_goal][self.est_other_goal]

    def policy_update(self):
        #現在のお互いの目的に沿ったQテーブルを返す
        self.policy = self.Q[self.my_goal][self.est_other_goal]

    def seed_reset(self):
        self.Q = []
        for i in range(self.number_of_goals):
            self.Q.append([])
        for i in range(self.number_of_goals):
            for j in range(self.number_of_goals):
                self.Q[i].append(defaultdict(lambda: [0] * len(self.actions)))

        self.infer_steps = 0
        self.estimation_value_other_goal = np.zeros(self.number_of_goals)
        self.estimation_score = np.zeros(self.number_of_goals)

    def reset(self):
        #エピソード開始時用初期化関数
        self.est_other_goal = np.random.randint(self.number_of_goals)
        self.my_goal = np.random.randint(self.number_of_goals)

        self.estimation_value_other_goal = np.zeros(self.number_of_goals)
        self.estimation_score = np.zeros(self.number_of_goals)

        self.policy_update()
        self.infer_steps = 0


    def greedy_act(self, now_state):
        #イプシロングリーディ
        n = now_state
        if np.random.random() < self.epsilon:
            # random
            return trans_ntoa(np.random.randint(len(self.actions)))
        else:
            # 現時点の良い行動を選択
            if n.repr() in self.policy and sum(self.policy[n.repr()]) != 0:
                return trans_ntoa(np.argmax(self.policy[n.repr()]))
            else:
                return trans_ntoa(np.random.randint(len(self.actions)))
        
    def learn(self, s, n_s, action, reward):
        a = trans_aton(action)
        if a != 4:
            gain = reward + self.gamma * max(self.Q[self.my_goal][self.est_other_goal][n_s.repr()])
            estimated = self.Q[self.my_goal][self.est_other_goal][s.repr()][a]
            self.Q[self.my_goal][self.est_other_goal][s.repr()][a] += self.alpha * (gain - estimated)

    def estimate_other_goal(self, other_state, other_action):
        other_action = trans_aton(other_action)
        if other_action != 4:
            self.infer_steps += 1
            policy_each_goal = []
            for other_goal in range(self.number_of_goals):
                if not (0 in self.Q[other_goal][self.my_goal][other_state.repr()]):
                    policy_each_goal.append(self.Q[other_goal][self.my_goal][other_state.repr()][other_action]/ sum(self.Q[other_goal][self.my_goal][other_state.repr()]))
                else :
                    policy_each_goal.append(1/self.number_of_goals)
            self.estimation_value_other_goal += policy_each_goal
            self.estimation_score = self.estimation_value_other_goal / self.infer_steps

            #update est_other_goal
            a = np.where(self.estimation_value_other_goal == max(self.estimation_value_other_goal))[0]
            self.est_other_goal = np.random.choice(a)

            #policy update
            self.policy_update()

        return self.est_other_goal


    def set_my_goal(self, goal_number):
        self.my_goal = goal_number
        self.policy_update()

    def get_my_goal(self):
        return self.my_goal

    def get_est_other_goal(self):
        return self.est_other_goal


class SOMQLearningAgent_Gchange(SOMQLearningAgent):
    def __init__(self, env, number_of_goals, epsilon=0.1, gamma=0.9, alpha=0.1, g_change_rate=0.5):
        super().__init__(env, number_of_goals, epsilon, gamma, alpha)

        self.change_rate=g_change_rate

    def reset(self, ini_state):
        self.est_other_goal = np.random.randint(self.number_of_goals)
        self.my_goal = np.random.randint(self.number_of_goals)
        if self.my_goal == 0:
            self.my_goal_state = State(0,7)
        elif self.my_goal == 1:
            self.my_goal_state = State(7,7)
        elif self.my_goal == 2:
            self.my_goal_state = State(7,0)
        elif self.my_goal == 3:
            self.my_goal_state = State(0,0)

        self.estimation_value_other_goal = np.zeros(self.number_of_goals)
        self.estimation_score = np.zeros(self.number_of_goals)

        self.policy_update()
        self.infer_steps = 0

        self.is_change = 0
        self.calcurate_init_distance_to_goal(ini_state)

    #about goal change
    def calcurate_init_distance_to_goal(self, ini_state):
        ini_row, ini_column = ini_state.row, ini_state.column
        g_row, g_column = self.my_goal_state.row, self.my_goal_state.column
        
        self.init_distance_to_goal = abs(ini_row-g_row) + abs(ini_column - g_column)

    def is_goal_change(self, n_state):
        if self.is_change:
            return False

        n_row, n_column = n_state.row, n_state.column
        g_row, g_column = self.my_goal_state.row, self.my_goal_state.column

        distance = abs(n_row - g_row) + abs(n_column - g_column)

        return distance < (self.init_distance_to_goal*self.change_rate)
        
        
    def goal_change(self, now_state):
        if self.is_goal_change(now_state):
            self.my_goal = np.random.randint(self.number_of_goals)
            self.is_change = 1
            self.policy_update()
            
        return self.my_goal

        

        