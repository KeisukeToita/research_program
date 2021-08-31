#environment関連のクラス

#ライブラリimport
import numpy as np
import copy
from agent import *


#environment関連のクラス

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
    def equal(self, other):
        return self.row == other.row and self.column == other.column

# 行動の定義

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

"""
迷路の情報
0:通路
9:壁
1~8:報酬用 reward_func関数にて設定
テキストファイル形式
"""
class Maze():

    def __init__(self, grid, init_agents_state=None, agent_num=1, is_goal=False):

        #about maze & agent
        self.grid = grid
        self.agent_num = agent_num
        self.agents_state = []
        for i in range(agent_num):
            self.agents_state.append(State())

        self.final_agents_state=[]#1エピソード終了後のエージェントを幽霊化させるための状態
        for i in range(agent_num):
            self.final_agents_state.append(State(99, i+1))

        self.now_agent = 0
        self.init_agents_state = init_agents_state

        #about agents goal
        self.is_goal=is_goal
        self.agents_goal=[]

        #about reward
        self.default_reward = 0
        self.colision_reward = -10 #マルチエージェントの時に使用
    
    @property
    def row_length(self):
        return len(self.grid)
    @property
    def column_length(self):
        return len(self.grid[0])
    @property
    def actions(self):
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY]

    #環境の初期化を行う
    def reset(self):
        #init state for random
        for i in range(self.agent_num):
            self.init_agents_state[i]=State(np.random.randint(self.row_length-2)+1, np.random.randint(self.column_length-2)+1)

        #copy and return
        self.agents_state = copy.deepcopy(self.init_agents_state) #エージェントの位置を初期化 *とりまこれだけ
        return copy.deepcopy(self.agents_state)
    
    def set_agents_goal(self, agents_goal):
        self.agents_goal = agents_goal
        
    #遷移のための関数    return 遷移確率
    def transit_func(self, state, action):
        transition_probs = {}
        #動けないときは空の辞書
        if not self.can_action_at(state):
            return transition_probs

        for a in self.actions:
            prob = 0
            #選んだ行動を確実にとる
            if a == action:
                prob = 1
            
            next_state = self._move(state, a)
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob
        return transition_probs
    
    #遷移を行う
    def transit(self, agent_number, state, action): #遷移確率をエージェントから獲得する．
        transition_probs = self.transit_func(state, action)
        if len(transition_probs) == 0:
            return None, None, True

        next_states = []
        probs = []

        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])

        #おそらく選択行動がそのまま反映されるはず
        next_state = np.random.choice(next_states, p=probs)

        #reward_func (goal or not)
        if self.is_goal:
            reward, done = self.reward_func_with_goal(agent_number, next_state)
        else:
            reward, done = self.reward_func(next_state)
        
        return next_state, reward, done
    
    #one step process
    def step(self, agent_number, action):
        self._set_now_agent(agent_number)
        next_state, reward, done = self.transit(agent_number, self.agents_state[self.now_agent], action)
        
        if next_state is not None:
            self.agents_state[self.now_agent] = next_state
            
        return next_state, reward, done

    #行動可能かの判定
    def can_action_at(self, state):
        #現在空マスにいるならTrue
        if self.grid[state.row][state.column] != -1:
            return True
        else:
            return False

    def _move(self, state, action):

        if not self.can_action_at(state):
            raise Exception("Can't move from here")

        next_state = state.clone()

        #移動
        if action == Action.UP:
            next_state.row -= 1
        if action == Action.DOWN:
            next_state.row += 1
        if action == Action.LEFT:
            next_state.column -= 1
        if action == Action.RIGHT:
            next_state.column += 1
        
        #移動可能かのチェック 無理なら元に戻す
        #迷路外に出たか
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        #壁にいるか
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    #報酬を与える関数
    def reward_func(self, state):
        reward = self.default_reward
        done = False

        attribute = self.grid[state.row][state.column]

        if attribute == 1:
            reward = 1000
            done = True
        elif attribute == -1:
            reward = -10
            done = True

        return reward, done

    def reward_func_with_goal(self, agent_number, state):
        reward = self.default_reward
        done = False

        attribute = self.grid[state.row][state.column]

        if attribute-1 == self.agents_goal[agent_number]:
            reward = 10000
            done = True

        return reward, done

    def colision_judge(self, next_states, rewards, dones):#衝突判定 衝突している者たちの報酬と
        n_s2 = copy.deepcopy(next_states)
        for i in range(len(next_states)):
            for j in range(len(n_s2)):
                if i != j and next_states[i].equal(n_s2[j]):#衝突しているエージェントについて書き換える
                    rewards[i] = self.colision_reward
                    dones[i] = True
        return  rewards, dones

    def _set_now_agent(self, number):
        self.now_agent = number

    def get_finish_state(self, number):
        return self.final_agents_state[number]