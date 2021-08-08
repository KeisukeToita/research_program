#environment関連のクラス

#ライブラリimport
import numpy as np

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
    def __eq__(self, other):
        return self.row == other.row and self.column == other.column

# 行動の定義

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


class Maze():

    def __init__(self, grid, init_agent_state=None, agent_num=1):

        self.grid = grid
        self.agent_num = agent_num
        self.agents_state = []
        for i in range(agent_num):
            self.agents_state.append(State())

        self.init_agent_state = State(3,0) #初期位置
        self.now_agent=0

        self.default_reward = -0.04
        self.collision_reward = -10 #マルチエージェントの時に使用
    
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
        for i in range(self.agent_num):
            self.agents_state[i]=self.init_agent_state #エージェントの位置を初期化 *とりまこれだけ
        return self.init_agent_state
        
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
    def transit(self, state, action): #遷移確率をエージェントから獲得する．
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

        #報酬獲得と終了判定
        reward, done = self.reward_func(next_state)
        return next_state, reward, done
    
    #1 step turn
    def step(self, agent_number, action):
        #TODO
        self._set_now_agent(agent_number)
        next_state, reward, done = self.transit(self.agents_state[self.now_agent], action)
        
        if next_state is not None:
            self.agents_state[self.now_agent] = next_state
            
        return next_state, reward, done

    #行動可能かの判定
    def can_action_at(self, state):
        #現在空マスにいるならTrue
        if self.grid[state.row][state.column] == 0:
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
            reward = 10
            done = True
        elif attribute == -1:
            reward = -10
            done = True
        elif attribute == 2:
            reward = 1000
            done = True

        return reward, done

    def _set_now_agent(self, number):
        self.now_agent = number