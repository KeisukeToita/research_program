import random
from enum import Enum
import numpy as np
import math
from collections import defaultdict


# 状態を表すクラス
class State():
    def __init__(self, row=-1, column=-1):
        self.column = column
        self.row = row

    # 状態の表現
    def repr(self):
        return "<State:[{}, {}]>".format(self.row, self.column)
    # クローン生成

    def clone(self):
        return State(self.row, self.column)
    # ハッシュ型のクローン?

    def __hash__(self):
        return hash((self.row, self.column))

    # 同値判定
    def __eq__(self, other):
        return self.row == other.row and self.column == other.column

# 行動の定義


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

# Agent_Base


class Agent():
    def __init__(self, env):
        self.observer = Observer(env)
        self.actions = env.actions

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
            if now_state in self.Q and sum(self.Q[now_state]) != 0:
                return trans_ntoa(np.argmax(self.Q[now_state]))
            else:
                return trans_ntoa(np.random.randint(len(self.actions)))

    def reward_add(self, reward):
        #reward_logに1エピソード分の報酬を追加
        self.reward_log.append(reward)

    def learn(self, s, n_s, action, reward):
        a = trans_aton(action)
        gain = reward + self.gamma * max(self.Q[n_s])
        estimated = self.Q[s][a]
        self.Q[s][a] += self.alpha * (gain - estimated)

            

def trans_ntoa(action_n):
    if action_n == 0:
        return Action.UP
    if action_n == 1:
        return Action.DOWN
    if action_n == 2:
        return Action.LEFT
    if action_n == 3:
        return Action.RIGHT
    if action_n == 4:
        return Action.STAY

def trans_aton(action):
    if action == Action.UP:
        return 0
    if action == Action.DOWN:
        return 1
    if action == Action.LEFT:
        return 2
    if action == Action.RIGHT:
        return 3
    if action == Action.STAY:
        return 4

# 環境の情報取得のためのクラス


class Observer():

    def __init__(self, env):
        self.env = env

    def get_state(self):  # 状態観測
        return self.env.agent_state

    def reset(self):
        # TODO
        pass

    def transform(self, state):  # 観測情報扱いやすい形に変換する
        # TODO
        pass


class Planner():

    def __init__(self):
        pass

    def learn(self, state, action, reward):  # 学習方法
        pass

    def policy(self):  # 行動選択
        pass


class LogShow():  # 記録用クラス

    def __init__(self, env):
        self.env = env
        pass

    def log_func(self):
        pass

    def show_q_value(self, Q):
        nrow = self.env.row_length
        ncol = self.env.column_length
