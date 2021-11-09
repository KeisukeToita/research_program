import gym
import numpy as np
import copy

"""
State クラス
状態単体のクラス
"""
class State:
    def __init__(self, x=-1, y=-1, angle=0, speed = 1):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
    
    def __repr__(self):
        return "<State: ({}, {}) {}rad>".format(self.x, self.y, self.angle)

    def clone(self):
        return State(self.x, self.y, self.angle)

    def eqpos(self, other):
        return self.x == other.x and self.y == other.y

"""
Multi State クラス
すべての状態を管理し，加工まで行う
"""
class Multi_State:
    def __init__(self, agent_num):
        self.agent_num = agent_num
        self.all_states = []
        for i in range(agent_num):
            self.all_states.append(State())

    def reset(self):
        pass

    def access(self, number):
        s = copy.deepcopy(self.all_states[number])
        return s

    """
    set関数
    """
    def set_value(self, agent_number, x, y, angle):
        self.all_states[agent_number].x = x
        self.all_states[agent_number].y = y
        self.all_states[agent_number].angle = angle
    
    def set_state(self, agent_number, state):
        self.all_states[agent_number] = state

    """
    表示用関数
    """
    def all_repr(self):
        for i in range(self.agent_num):
            print("agent {}:".format(i+1) + self.all_states[i].repr())

    """
    各エージェントのother_statesを生成し，まとめて返す．
    """
    def get_other_states(self):
        pass

"""
Maze クラス
Open AI Gym の継承クラス

ルールを以下に記載：

・obs ・・・　次状態

・_single_unit ・・・各エージェント用の処理

"""
class Maze(gym.Env):

    def __init__(self, agent_num):
        """
        必要なデータ
        
        エージェント数に応じた状態数の用意
        報酬
        """
        self.four_sides = {}
        self.four_sides["top"] = self.four_sides["right"] = 10
        self.four_sides["bottom"] = self.four_sides["left"] = 0

        self.multi_agent_state = Multi_State(agent_num)

        self.agent_num = agent_num
        
        self.default_reward = -0.05

    def _reset(self):
        """
        環境の状態の初期化
        """
        pass

    def _step(self, action):
        """
        受け取ったアクションに対して１対１体のエージェントを動かしていく
        """
        """
        TODO 現在の状態をチェックして，終了になっているものの処理をする
        """
        # 遷移
        obs, reward, done = self.transit(self.multi_agent_state, action)

        # 次状態をチェック & 更新
        for i in range(self.agent_num):
            if obs.access(i) is not None:
                self.multi_agent_state.set_state(i, obs.access(i))

        return obs, reward, done, None #== info


    def _render(self, mode='human', close=False):
        self.multi_agent_state.all_repr()

    def _seed(self, seed=None):
        np.random.seed(seed)

    """
    遷移用の関数
    """
    def transit(self, multi_state, action):
        pass

    def move(self, multi_state, action):        
        next_multi_state = copy.deepcopy(multi_state) #全体状態のクローン

        for i in range(self.agent_num):
            next_state = self.move_single_unit(next_multi_state.access(i), action[i])
            next_multi_state.set_state(next_state)

        return next_multi_state

    def move_single_unit(self, state, a):
        next_state = state.clone() #状態のクローン

        # angle の更新
        next_state.angle += a
        if next_state.angle >= 360:
            next_state.angle -= 360
        elif next_state.angle < 0:
            next_state.angle += 360

        # 差分の導出
        # x_move = next_state.speed * cos(next_state.angle)
        # y_move = next_state.speed * sin(next_state.angle)

        # 差分を加え，次状態にする
        # next_state.x += x_move
        # next_state.y += y_move

        if self.is_area_out(next_state):
            next_state = state

        return next_state
    
    # 動けるのかどうかの判定
    def can_action_at(self, state):
        pass

    # エリア外に出たかどうかの判定
    def is_area_out(self, state):
        if not(self.four_sides["bottom"] <= state.y < self.four_sides["top"]):
            return True
        if not(self.four_sides["left"] <= state.x < self.four_sides["right"]):
            return True
        return False

    """
    報酬用の関数
    """
    def reward_func(self, multi_state):
        # 単体の報酬を取得
        reward_list = []
        for i in range(self.agent_num):
            reward = self.reward_func_single_unit(multi_state.access(i))
            reward_list.append(reward)

        # 衝突判定
        reward_list = self.is_colision(multi_state, reward_list)

        return reward_list

    def reward_func_single_unit(self, state):
        pass

    # 衝突判定の関数
    def is_colision(self, multi_state, reward_list):
        pass
