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
    def __init__(self, env, planner):
        self.observer = Observer(env)
        self.planner = planner
        self.actions = env.actions()

    def act(self): #TODO 行動を返す関数
        
        return 0

    def learn(self): #TODO 学習方法について(引数には必要な要素)
        pass
        
#環境の情報取得のためのクラス
class Observer():

    def __init__(self, env):
        self.env = env

    def get_state(self):
        return self.env.agent_state

    def reset(self):
        #TODO
        pass

    def transform(self, state):
        #TODO
        pass
