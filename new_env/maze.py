import gym
import numpy as np
import copy
import math

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
    
    def _repr(self):
        return "<State: ({}, {}) {}rad>".format(self.x, self.y, int(self.angle))

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
    def clone(self):
        m_s = Multi_State(self.agent_num)
        for i in range(self.agent_num):
            m_s.set_state(i, self.all_states[i].clone())
        return m_s

    """
    set関数
    """
    def set_value(self, agent_number, x, y, angle):
        self.all_states[agent_number].x = x
        self.all_states[agent_number].y = y
        self.all_states[agent_number].angle = angle
    
    def set_state(self, agent_number, state):
        self.all_states[agent_number] = state
        
    def exit_the_state(self, state):
        for i in range(self.agent_num):
            if self.all_states[i].eqpos(state):
                return True
    """
    表示用関数
    """
    def all_repr(self):
        for i in range(self.agent_num):
            print("agent {}:".format(i+1) + self.all_states[i]._repr())

    """
    各エージェントのother_statesを生成し，まとめて返す．
    """
    def get_other_states(self):
        other_states_list = []
        for i in range(self.agent_num):
            o_s = [x for j,x in enumerate(self.all_states) if i!=j]
            other_states_list.append(o_s)
        return other_states_list

class Finish_State(Multi_State):
    OUT_POSITION = 99
    def __init__(self, agent_num):
        super().__init__(agent_num)
        for i in range(agent_num):
            self.set_state(i, State(self.OUT_POSITION, i))
"""
Maze クラス
Open AI Gym の継承クラス

ルールを以下に記載：

・obs ・・・　次状態

・_single_unit ・・・各エージェント用の処理
"""
"""
各エージェントが取る状態：Stateクラス
                    座標(x, y)と向いてる方向angle, 
                    その方向にどの程度の速さで進んでいるかを示すspeed
受け付けるアクション：各エージェントの行動をまとめたリスト
返す報酬：各エージェントに対する報酬をまとめたリスト
"""

class Maze(gym.Env):  
    """
    gym環境に沿った関数
    """
    def __init__(self, agent_num, area_conf, reset_conf=None, action_conf=None):
        """
        必要なデータ
        """
        #エリアの設定
        self.area_conf = area_conf
        #エージェントの状態(マルチ)
        self.agent_num = agent_num
        self.multi_agent_state = Multi_State(agent_num)
        #行動空間(とりあえず離散でやらせてください)
        self.action_space = gym.spaces.Discrete(action_conf.ACTION_NUM)
        self.action_list = action_conf.action_list_make()
        #ゴール設定
        self.agents_goal = []
        #終了判定用
        self.dones = np.zeros(agent_num)
        self.colisions = np.zeros(agent_num)
        #エージェント数
        self.agent_num = agent_num
        self.agent_size = 1
        #報酬系
        self.default_reward = 0
        self.goal_reward = 10
        self.colision_reward = -10
        self.finish_reward = 0
        #各エリア判定用クラス
        self.maze_zone = Maze_Zone(area_conf)
        #初期のデータ (エージェントの配置，各ゴールなど)
        self.reset_conf = reset_conf
        #終了後の状態
        self.finish_state = Finish_State(agent_num)

    def _reset(self):
        """
        エージェントの状態の初期化
        ゴールも初期化する
        """
        #終了判定用
        self.dones = np.zeros(self.agent_num)
        self.colisions = np.zeros(self.agent_num)
        #初期位置のデータがあるとき
        if self.reset_conf.ini_state_data != None:
            self.multi_agent_state = self.reset_conf.ini_state_data
        else:
            self.multi_agent_state = Multi_State(self.agent_num)
            i=0
            while i < self.agent_num:
                x = np.random.randint(1,self.area_conf.WIDTH)
                y = np.random.randint(1,self.area_conf.HEIGHT)
                angle = np.random.randint(0,3)*90
                if not self.multi_agent_state.exit_the_state(State(x, y, angle)):
                    self.multi_agent_state.set_state(i, State(x, y, angle))
                    i += 1
        
        #ゴール設定のデータがある時
        if self.reset_conf.ini_goal_data != None:
            self.agents_goal = self.reset_conf.ini_goal_data
        else:
            self.agents_goal = np.random.randint(self.area_conf.goal_num, size=self.agent_num)
            
        #deep copyして返す．
        mul_age_state = copy.deepcopy(self.multi_agent_state)
        age_goal = copy.deepcopy(self.agents_goal)
        return mul_age_state, age_goal

    def _step(self, actions):
        """
        受け取ったアクションに対して１対１体のエージェントを動かしていく
        """
        """
        TODO 現在の状態をチェックして，終了になっているものの処理をする
        """
        # 遷移
        obs, reward, done = self.transit(self.multi_agent_state, actions)

        # 次状態をチェック & 更新
        for i in range(self.agent_num):
            if obs.access(i) is not None:
                self.multi_agent_state.set_state(i, obs.access(i))

        return obs, reward, done, None #== info

    def _render(self, mode='human', close=False):
        #現在はマルチエージェントの状態を表示するだけ．
        self.multi_agent_state.all_repr()
        
        # area = np.zeros((self.area_conf.HEIGHT, self.area_conf.WIDTH))
        
        # for i in range(self.agent_num):
        #     s = self.multi_agent_state.access(i)
        #     f_s = self.finish_state.access(i)
        #     if not f_s.eqpos(s):
        #         x, y = int(s.x), int(s.y)
        #         area[y][x] += i+1
            
        # area = np.flipud(area)
        # for i in range(self.area_conf.HEIGHT):
        #     print(area[i])

    def _seed(self, seed=None):
        np.random.seed(seed)

    def action_sample(self):
        action_list = np.random.randint(0, self.action_space.n, self.agent_num)
        return action_list
    """
    遷移用の関数
    """
    def transit(self, multi_state, actions):
        
        #遷移
        next_multi_state = self.move(multi_state, actions)
        #次状態における報酬の獲得
        reward_list = self.reward_func(next_multi_state)
        #終了判定
        done = False not in self.dones
        
        return next_multi_state, reward_list, done
    
    def transit_func(self, multi_state, actions):
        """
        冒頭，遷移確率を考慮するための関数
        現コードでは,使用していないが，いずれ必要になる可能性あるよ．
        """
        pass

    def move(self, multi_state, actions):        
        next_multi_state = multi_state.clone() #全体状態のクローン

        for i in range(self.agent_num):
            if self.dones[i]==False:
                next_state = self.move_single_unit(next_multi_state.access(i), actions[i])
            elif self.dones[i]==True and self.colisions[i]==True:
                next_state = next_multi_state.access(i)
            elif self.dones[i]==True and self.colisions[i]==False:
                next_state = self.finish_state.access(i)
                
            next_multi_state.set_state(i, next_state)

        return next_multi_state

    def move_single_unit(self, state, a):
        next_state = state.clone() #状態のクローン
        angle = self.action_list[a]

        # angle の更新
        n_ang = next_state.angle + angle
        if n_ang >= 360:
            n_ang -= 360
        elif n_ang < 0:
            n_ang += 360
        next_state.angle = n_ang

        # 差分の導出
        rad_ang = math.radians(next_state.angle)
        x_move = next_state.speed * math.cos(rad_ang)
        y_move = next_state.speed * math.sin(rad_ang)

        # 差分を加え，次状態にする
        next_state.x += x_move
        next_state.y += y_move

        if self.maze_zone.is_area_out(next_state):
            #TODO 角度は変えたい．
            next_state = state.clone()
            next_state.angle = n_ang
        return next_state
    # 動けるのかどうかの判定
    def can_action_at(self, number):
        #終了してたら動けない
        return not self.dones[number]

    """
    報酬用の関数
    """
    def reward_func(self, multi_state):
        
        # 単体の報酬を取得 (ゴール判定)
        reward_list = []
        for i in range(self.agent_num):
            if not self.dones[i]:
                goal = self.agents_goal[i]
                reward, done = self.reward_func_single_unit(multi_state.access(i), goal)
                reward_list.append(reward)
                self.dones[i] = done
            else:
                reward_list.append(self.finish_reward)

        # 衝突判定
        #reward_list = self.is_colision(multi_state, reward_list)

        return reward_list

    def reward_func_single_unit(self, state, goal_number):
        reward = self.default_reward
        done = False
        
        #ゴールに到達したかどうかのチェック
        if self.maze_zone.achieve_goal(state, goal_number):
            reward = self.goal_reward
            done = True
        
        return reward, done

    """
    衝突判定
    """
    def is_colision(self, multi_state, reward_list):
        for i in range(self.agent_num):
            for j in range(1, self.agent_num):
                state_a = multi_state.access(i)
                state_b = multi_state.access(j)
                if self.is_colision_one_pair(state_a, state_b) and (i != j):
                    if self.dones[i] == False and self.dones[j] == False:
                        reward_list[i] = reward_list[j] = self.colision_reward
                        self.colisions[i] = self.colisions[j] = True
                        self.dones[i] = self.dones[j] = True
                    elif self.dones[i] == True and self.dones[j] == False:
                        if self.colisions[i] == True:
                            self.dones[j] = True
                            self.colisions[j] = True
                            reward_list[i] = self.finish_reward
                            reward_list[j] = self.colision_reward
                    elif self.dones[i] == False and self.dones[j] == True:
                        if self.colisions[j] == True:
                            self.dones[i] = True
                            self.colisions[i] = True
                            reward_list[j] = self.finish_reward
                            reward_list[i] = self.colision_reward
        return reward_list
    #2つの状態を取得し，重なっているかどうかを判定する関数
    def is_colision_one_pair(self, s_a, s_b):
        a = np.array((s_a.x, s_a.y))
        b = np.array((s_b.x, s_b.y))
        
        dist = np.linalg.norm(a-b)
        
        return dist <= self.agent_size

"""
Maze_Zone
エリア判定担当のクラス
"""
class Maze_Zone:
    
    def __init__(self, area_conf):
        self.area_conf = area_conf
        
        self.area_width, self.area_height = area_conf.WIDTH, area_conf.HEIGHT
        
    # エリア外に出たかどうかの判定
    def is_area_out(self, state):
        if not(0 <= state.y < self.area_height):
            return True
        if not(0 <= state.x < self.area_width):
            return True
        return False
    
    #ゴール判定
    def achieve_goal(self, state, goal_number):
        #ゴール到達の判定
        x, y = state.x, state.y
        #get goal_data
        g_data = self.area_conf.goal_data(goal_number)
        
        #範囲外ならFalse
        if not(g_data["x_min"] <= x <= g_data["x_max"]):
            return False
        if not(g_data["y_min"] <= y <= g_data["y_max"]):
            return False
        return True
    