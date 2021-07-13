import random
from enum import Enum
import numpy as np

#状態を表すクラス
class State():
    def __init__(self, row=-1, column=-1):
        self.column = column
        self.row = row

    #状態の表現
    def __repr__(self):
        return "<State:[{}, {}]>".format(self.row, self.column)
    #クローン生成
    def clone(self):
        return State(self.row, self.column)
    #ハッシュ型のクローン?
    def __hash__(self):
        return hash((self.row, self.column))

    #同値判定
    def __eq__(self, other):
        return self.row == other.row and self.column == other.column

#行動の定義
class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2
    STAY = 0

#Agent_Base
class Agent():
    def __init__(self, env):
        self.observer = Observer(env)
        #self.planner = planner
        self.actions = env.actions()

    def act(self): #TODO 行動を返す関数 & Plannerに行動選択も委託しよう
        a = random.choice(self.actions)
        return a

    def learn(self): #TODO 学習方法について(引数には必要な要素) & Plannerに役割を分散できるか．
        pass
        
#環境の情報取得のためのクラス
class Observer():

    def __init__(self, env):
        self.env = env

    def get_state(self): #状態観測
        return self.env.agent_state

    def reset(self): 
        #TODO
        pass

    def transform(self, state): #観測情報扱いやすい形に変換する
        #TODO
        pass

class Planner():

    def __init__(self):
        pass
    
    def learn(self, state, action, reward): #学習方法
        pass

    def policy(self): #行動選択
        pass

class Logger(): #記録用クラス

    def __init__(self):
        pass

    def log_func(self):
        pass