import random
from enum import Enum
import numpy as np
import math
import copy

from base_utils import *
from collections import defaultdict
from maze_8direction import *

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

class Action(Enum):#TODO 各所８方向 & STAY番号を８に変更
    U = 0 #UP
    UR = 1 #UP&RIGHT
    R = 2 #RIGHT
    DR = 3 #DOWN&RIGHT
    D = 4 #DOWN
    DL = 5 #DOWN&LEFT
    L = 6 #LEFT
    UL = 7 #UP&LEFT
    S = 8 #STAY


class ActPlan:
    straight = 0
    go_right = 1
    go_left = 2
    no_needed = 3

class Base_Agent:
    def __init__(self, env, epsilon=0.1, gamma=0.9, alpha=0.1):

        self.actions = [Action.U,
                        Action.UR,
                        Action.R,
                        Action.DR,
                        Action.D,
                        Action.DL,
                        Action.L,
                        Action.UL]
        #Q_table
        self.Q = defaultdict(lambda: [0] * len(self.actions))

        #parameter
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        #policy
        self.policy = self.Q

    def seed_reset(self):
        self.Q = defaultdict(lambda: [0] * len(self.actions))
        self.policy_update()

    def episode_reset(self): #func of episode reset
        
        pass        

    def greedy_act(self, now_state): #epsilon greedy
        n = now_state
        if np.random.random() < self.epsilon:
            # random
            return trans_ntoa(np.random.randint(len(self.actions)))
        else:
            # choice best action
            if n.repr() in self.policy and sum(self.policy[n.repr()]) != 0:
                return trans_ntoa(np.argmax(self.policy[n.repr()]))
            else:
                return trans_ntoa(np.random.randint(len(self.actions)))
        
    def learn(self, s, n_s, action, reward):
        a = trans_aton(action)
        if a != 8:
            gain = reward + self.gamma * max(self.Q[n_s.repr()])
            estimated = self.Q[s.repr()][a]
            self.Q[s.repr()][a] += self.alpha * (gain - estimated)
        self.policy_update()

    def policy_update(self):
        self.policy = self.Q

class ActPlanAgent(Base_Agent):
    GOAL_MODE = 0
    ACTPLAN_MODE = 1

    EPSILON_GREEDY_MODE = "EPSILON_GREEDY"
    SOFTMAX_MODE = "SOFTMAX"

    #TODO 相対座標は序盤ランダム性があるので、一回計算したものをワンステップ通して使うように改変する必要あり。
    def __init__(self, env, number_of_goals, epsilon=0.1, gamma=0.9, alpha=0.1, act_mode="SOFTMAX"):
        super().__init__(env, epsilon, gamma, alpha)
        self.set_mode(self.GOAL_MODE) #仮設定
        
        self.act_mode = act_mode

        #about goal 
        self.number_of_goals = number_of_goals
        self.goalQ = []
        for i in range(number_of_goals):
            self.goalQ.append([])
        for i in range(number_of_goals):
            for j in range(number_of_goals):
                self.goalQ[i].append(defaultdict(lambda: [0] * len(self.actions)))

        self.my_goal = np.random.randint(number_of_goals)
        self.est_other_goal = np.random.randint(number_of_goals)

        # infer_part
        self.goal_infer_steps = 0
        self.goal_estimation_value = np.zeros(number_of_goals)
        self.goal_estimation_score = np.zeros(number_of_goals)  
        
        #about actplan
        self.actplans = [ActPlan.straight, ActPlan.go_right, ActPlan.go_left, ActPlan.no_needed]
        self.number_of_actplans = len(self.actplans)
        actplan_n = len(self.actplans)
        self.actplanQ = []
        for i in range(actplan_n):
            self.actplanQ.append([])
        for i in range(actplan_n):
            for j in range(actplan_n):
                self.actplanQ[i].append(defaultdict(lambda: [0] * len(self.actions)))

        self.my_actplan = np.random.randint(self.number_of_actplans)
        self.est_other_actplan = np.random.randint(self.number_of_actplans)

        # infer_part
        self.actplan_infer_steps = 0
        self.actplan_estimation_value = np.zeros(actplan_n)
        self.actplan_estimation_score = np.zeros(actplan_n)
        
        self.policy = self.goalQ[self.my_goal][self.est_other_goal]

        #about past action
        self.past_action = None
        self.other_past_action = None
    
    #about reset func
    def seed_reset(self):
        # #reset goalQ
        # self.goalQ = []
        # for i in range(self.number_of_goals):
        #     self.goalQ.append([])
        # for i in range(self.number_of_goals):
        #     for j in range(self.number_of_goals):
        #         self.goalQ[i].append(defaultdict(lambda: [0] * len(self.actions)))
        # #reset actplanQ
        # actplan_n = len(self.actplans)
        # self.actplanQ = []
        # for i in range(actplan_n):
        #     self.actplanQ.append([])
        # for i in range(actplan_n):
        #     for j in range(actplan_n):
        #         self.actplanQ[i].append(defaultdict(lambda: [0] * len(self.actions)))

        self.policy_update()

    def episode_reset(self):
        self.est_other_goal = np.random.randint(self.number_of_goals)
        self.my_goal = np.random.randint(self.number_of_goals)

        self.goal_estimation_value = np.zeros(self.number_of_goals)
        self.goal_estimation_score = np.zeros(self.number_of_goals)

        self.goal_infer_steps = 0
        self.policy_update()

        self.past_action = None


    def actplan_phase_reset(self):
        self.est_other_actplan = np.random.randint(len(self.actplans))
        self.actplan_update()

        self.actplan_estimation_value = np.zeros(len(self.actplans))
        self.actplan_estimation_score = np.zeros(len(self.actplans))

        self.actplan_infer_steps = 0

    #about action func
    def act(self, my_state, angle=None):
        if self.act_mode == self.EPSILON_GREEDY_MODE:
            return self.greedy_act(my_state, angle)
        elif self.act_mode == self.SOFTMAX_MODE:
            return self.softmax_act(my_state, angle)

    def greedy_act(self, my_state, angle=None):
        n = my_state
        if np.random.random() < self.epsilon:
            # random
            action = np.random.randint(len(self.actions))
        else:
            # choice best action
            if n.repr() in self.policy and sum(self.policy[n.repr()]) != 0:
                action = np.argmax(self.policy[n.repr()])
            else:
                action = np.random.randint(len(self.actions))

        if self.mode == self.ACTPLAN_MODE:
            action = self.get_rotate_action(action, angle)
        return trans_ntoa(action)
    
    def softmax_act(self, my_state, angle=None):
        n = my_state
        sum_exp_values = sum([np.exp(v) for v in self.policy[n.repr()]])
        p = [np.exp(v)/sum_exp_values for v in self.policy[n.repr()]]
        action = np.random.choice(np.arange(len(self.actions)), p=p)
        if self.mode == self.ACTPLAN_MODE:
            action = self.get_rotate_action(action, angle)
        return trans_ntoa(action)

    #about get relative state funcs
    def get_my_direction(self, my_state):
        m = my_state
        if m.repr() in self.goalQ[self.my_goal][self.est_other_goal] and sum(self.goalQ[self.my_goal][self.est_other_goal][m.repr()]) != 0:
            return np.argmax(self.goalQ[self.my_goal][self.est_other_goal][m.repr()])
        if self.past_action != None:
            return self.past_action
        return np.random.randint(len(self.actions))
    def get_other_direction(self):#need observe other past action
        if self.other_past_action != None:
            return self.other_past_action
        return np.random.randint(len(self.actions))
    def get_relative_state(self, my_state, other_state, n_Dir=None):
        m_r = my_state.row
        m_c = my_state.column
        o_c = other_state.column
        o_r = other_state.row
        #relative column & relative row
        r_r, r_c = o_r-m_r, o_c-m_c
        #direction
        if n_Dir == None:
            Dir = self.get_my_direction(my_state)
        else:
            Dir = n_Dir
        #rotate    
        if Dir == 0 or Dir == 1:
            angle = 0
        elif Dir == 2 or Dir == 3:
            r_r, r_c = r_c*-1, r_r
            angle = 90
        elif Dir == 4 or Dir == 5:
            r_r, r_c = r_r*-1, r_c*-1
            angle = 180
        elif Dir == 6 or Dir == 7:
            r_r, r_c = r_c, r_r*-1
            angle = 270
        else:
            angle = 0

        return State(r_r, r_c), angle
    def get_rotate_action(self, action, angle): #return action_N
        action = trans_aton(action)
        #action rotate
        r_action = action + (angle / 45)
        #fix of action
        if r_action >= 8:
            r_action -= 8
        elif r_action < 0:
            r_action += 8

        return int(r_action)
    
    #about learn func
    def learn(self, s, n_s, action, reward, angle=None):
        a = trans_aton(action)
        if a != 8:
            if self.mode == self.GOAL_MODE:
                gain = reward + self.gamma * max(self.goalQ[self.my_goal][self.est_other_goal][n_s.repr()])
                estimated = self.goalQ[self.my_goal][self.est_other_goal][s.repr()][a]
                self.goalQ[self.my_goal][self.est_other_goal][s.repr()][a] += self.alpha * (gain - estimated)
            elif self.mode == self.ACTPLAN_MODE:
                if reward > 0:
                    self.get_rotate_action(a, angle*-1)
                    gain = reward + self.gamma * max(self.goalQ[self.my_goal][self.est_other_goal][n_s.repr()])
                    estimated = self.goalQ[self.my_goal][self.est_other_goal][s.repr()][a]
                    self.goalQ[self.my_goal][self.est_other_goal][s.repr()][a] += self.alpha * (gain - estimated)
                elif reward <= 0:
                    gain = reward + self.gamma * max(self.actplanQ[self.my_actplan][self.est_other_actplan][n_s.repr()])
                    estimated = self.actplanQ[self.my_actplan][self.est_other_actplan][s.repr()][action]
                    self.actplanQ[self.my_actplan][self.est_other_actplan][s.repr()][action] += self.alpha * (gain - estimated)
        self.policy_update()

    #about estimate func
    def estimate_other_goal(self, other_state, other_action):
        other_action = trans_aton(other_action)
        if other_action != 8:
            self.goal_infer_steps += 1
            policy_each_goal = []
            for other_goal in range(self.number_of_goals):
                if not (0 in self.goalQ[other_goal][self.my_goal][other_state.repr()]):
                    policy_each_goal.append(self.goalQ[other_goal][self.my_goal][other_state.repr()][other_action]/ sum(self.goalQ[other_goal][self.my_goal][other_state.repr()]))
                else:
                    policy_each_goal.append(1/self.number_of_goals)
            self.goal_estimation_value += policy_each_goal
            self.goal_estimation_score = self.goal_estimation_value / self.goal_infer_steps

            #update est_other_goal
            a = np.where(self.goal_estimation_value == max(self.goal_estimation_value))[0]
            self.est_other_goal = np.random.choice(a)

            #policy update
            self.policy_update()

        return self.est_other_goal
    
    def estimate_other_actplan(self, other_state, other_action):
        other_action = trans_aton(other_action)
        if other_action != 8:
            #TODO infer part fix
            self.actplan_infer_steps += 1
            policy_each_actplan = []
            for other_actplan in range(len(self.actplans)):
                if not (0 in self.actplanQ[other_actplan][self.my_goal][other_state.repr()]):
                    policy_each_actplan.append(self.goalQ[other_actplan][self.my_goal][other_state.repr()][other_action]/ sum(self.goalQ[other_actplan][self.my_goal][other_state.repr()]))
                else:
                    policy_each_actplan.append(1/len(self.actplans))
            self.actplan_estimation_value += policy_each_actplan
            self.actplan_estimation_score = self.actplan_estimation_value / self.actplan_infer_steps

            #update est_other_goal
            a = np.where(self.actplan_estimation_value == max(self.actplan_estimation_value))[0]
            self.est_other_actplan = np.random.choice(a)

            #policy update
            self.policy_update()

        return self.est_other_actplan

    #about update_func
    def mode_update(self, my_state, other_state):
        # m_row = my_state.row
        # m_col = my_state.column
        # #mode judge
        # mode = self.GOAL_MODE
        # for x in range(-2, 3):
        #     for y in range(-2, 3):
        #         if other_state.repr() == State(m_row+y, m_col+x).repr():
        #             mode = self.ACTPLAN_MODE
        mode = self.GOAL_MODE
        #mode set
        self.mode = mode
    def policy_update(self):
        if self.mode == self.GOAL_MODE:
            self.policy = self.goalQ[self.my_goal][self.est_other_goal]
        elif self.mode == self.ACTPLAN_MODE:
            self.policy = self.actplanQ[self.my_actplan][self.est_other_actplan]
    def actplan_update(self):
        #TODO Fix actplan update
        self.my_actplan = np.random.randint(self.number_of_actplans)

    #get func
    def get_mode(self):
        return self.mode
    def get_my_goal(self):
        return self.my_goal
    def get_my_actplan(self):
        return self.my_actplan
    def get_est_other_goal(self):
        return self.est_other_goal
    def get_est_other_actplan(self):
        return self.est_other_actplan
    #set func
    def set_past_action(self, action):
        self.past_action = action
    def set_mode(self, mode):
        self.mode = mode     


class ActPlanAgent_with_direction(ActPlanAgent):
    NOT_CHOICE_VALUE = -10

    def __init__(self, env, number_of_goals, epsilon, gamma, alpha, act_mode):
        super().__init__(env, number_of_goals, epsilon=epsilon, gamma=gamma, alpha=alpha, act_mode=act_mode)

        actplan_n = len(self.actplans)
        direction_n = 4 #up, down, left, right
        self.actplanQ = []
        for i in range(actplan_n):
            self.actplanQ.append([])
        for i in range(actplan_n):
            for j in range(direction_n):
                self.actplanQ[i].append(defaultdict(lambda: [0] * len(self.actions)))

        #about direction
        self.my_direction = np.random.randint(4)
        self.other_direction = np.random.randint(4)
        self.next_direction = None

    #reset func
    def actplan_phase_reset(self):
        self.est_other_actplan = np.random.randint(len(self.actplans))

        self.actplan_estimation_value = np.zeros(len(self.actplans))
        self.actplan_estimation_score = np.zeros(len(self.actplans))

        self.actplan_one_change = False

        self.actplan_infer_steps = 0

    def episode_reset(self, goal=None):
        self.est_other_goal = np.random.randint(self.number_of_goals)
        if goal == None:
            self.my_goal = np.random.randint(self.number_of_goals)
        else:
            self.my_goal = goal

        self.goal_estimation_value = np.zeros(self.number_of_goals)
        self.goal_estimation_score = np.zeros(self.number_of_goals)

        self.goal_infer_steps = 0
        self.policy_update()

        self.past_action = None

        self.est_other_actplan = np.random.randint(len(self.actplans))

        self.actplan_estimation_value = np.zeros(len(self.actplans))
        self.actplan_estimation_score = np.zeros(len(self.actplans))

        self.actplan_one_change = False

        self.actplan_infer_steps = 0

    #about actplan func
    def make_actplan_policy(self, policy, actplan):
        #TODO 行動方針に応じたpolicyを出力するための関数
        actplan_policy = np.array(policy)

        if actplan == self.actplans[0]:#straight
            actplan_policy[2:7] = self.NOT_CHOICE_VALUE #UP ONLY
        elif actplan == self.actplans[1]:#right
            actplan_policy[3:8] = self.NOT_CHOICE_VALUE # UR&R ONLY
        elif actplan == self.actplans[2]:#left
            actplan_policy[1:6] = self.NOT_CHOICE_VALUE # UL&L ONLY
        elif actplan == self.actplans[3]:#no needed
            #目的用の方策にするか，縛らないのか，，どうしようかね，
            pass

        return actplan_policy

    #about action func
    def act(self, my_state, angle=None, dire=None):
        if self.act_mode == self.EPSILON_GREEDY_MODE:
            return self.greedy_act(my_state, angle, dire)
        elif self.act_mode == self.SOFTMAX_MODE:
            return self.softmax_act(my_state, angle, dire)

    def greedy_act(self, my_state, angle=None, dire=None):
        n = my_state
        
        #policy choice
        if self.mode == self.GOAL_MODE:
            the_policy = self.policy[n.repr()]
            choice_policy = self.policy
        elif self.mode == self.ACTPLAN_MODE:
            dire = self.get_rotate_direction(dire, angle*-1)
            the_policy = self.make_actplan_policy(self.policy[dire][n.repr()], self.my_actplan)
            choice_policy = self.policy[dire]

        #action choice
        if np.random.random() < self.epsilon:
            # random
            action = np.random.randint(len(self.actions))
        else:
            # choice best action
            if n.repr() in choice_policy and sum(the_policy) != 0:
                action = np.argmax(the_policy)
            else:
                action = np.random.randint(len(self.actions))

        if self.mode == self.ACTPLAN_MODE:
                action = self.get_rotate_action(action, angle)
        return trans_ntoa(action)
    
    def softmax_act(self, my_state, angle=None, dir=None):
        n = my_state
        #policy choice
        if self.mode == self.GOAL_MODE:
            the_policy = self.policy[n.repr()]
        elif self.mode == self.ACTPLAN_MODE:
            dir = self.get_rotate_direction(dir, angle*-1)
            the_policy = self.make_actplan_policy(self.policy[dir][n.repr()], self.my_actplan)
        
        #action choice
        sum_exp_values = sum([np.exp(v) for v in the_policy])
        p = [np.exp(v)/sum_exp_values for v in the_policy]
        action = np.random.choice(np.arange(len(self.actions)), p=p)

        if self.mode == self.ACTPLAN_MODE:
            action = self.get_rotate_action(action, angle)

        return trans_ntoa(action)

    #learn func
    def learn(self, s, n_s, action, reward, angle=None, dir=None, n_dir=None):
        a = trans_aton(action)
        if a != 8:
            if self.mode == self.GOAL_MODE:
                gain = reward + self.gamma * max(self.goalQ[self.my_goal][self.est_other_goal][n_s.repr()])
                estimated = self.goalQ[self.my_goal][self.est_other_goal][s.repr()][a]
                self.goalQ[self.my_goal][self.est_other_goal][s.repr()][a] += self.alpha * (gain - estimated)
            elif self.mode == self.ACTPLAN_MODE:
                if reward > 0:
                    #TODO s & n_s remake
                    self.get_rotate_action(a, angle*-1)
                    gain = reward + self.gamma * max(self.goalQ[self.my_goal][self.est_other_goal][n_s.repr()])
                    estimated = self.goalQ[self.my_goal][self.est_other_goal][s.repr()][a]
                    self.goalQ[self.my_goal][self.est_other_goal][s.repr()][a] += self.alpha * (gain - estimated)
                elif reward <= 0:
                    gain = reward + self.gamma * max(self.actplanQ[self.est_other_actplan][n_dir][n_s.repr()])
                    estimated = self.actplanQ[self.est_other_actplan][dir][s.repr()][action]
                    self.actplanQ[self.est_other_actplan][dir][s.repr()][action] += self.alpha * (gain - estimated)
        self.policy_update()

    #estimate_func
    def estimate_other_actplan(self, other_state, other_action, dire):
        other_action = trans_aton(other_action)
        if other_action != 8:
            self.actplan_infer_steps += 1
            #print(self.actplan_estimation_value)
            policy_each_actplan = []
            for other_actplan in range(len(self.actplans)):
                the_policy = self.make_actplan_policy(self.actplanQ[self.my_actplan][dire][other_state.repr()], other_actplan)
                the_policy = [np.exp(v) for v in the_policy]
                if sum(the_policy) != 0:
                    policy_each_actplan.append(the_policy[other_action]/ sum(the_policy))
                else:
                    policy_each_actplan.append(0) 
            self.actplan_estimation_value += policy_each_actplan
            self.actplan_estimation_score = self.actplan_estimation_value / self.actplan_infer_steps
            #update est_other_goal
            a = np.where(self.actplan_estimation_value == max(self.actplan_estimation_value))[0]
            self.est_other_actplan = np.random.choice(a)

            #policy update
            self.policy_update()

        return self.est_other_actplan

    #about direction func
    def decide_my_direction(self, my_state):
        m = my_state
        if self.next_direction == None:
            if m.repr() in self.goalQ[self.my_goal][self.est_other_goal] and sum(self.goalQ[self.my_goal][self.est_other_goal][m.repr()]) != 0:
                direction = np.argmax(self.goalQ[self.my_goal][self.est_other_goal][m.repr()])
            elif self.past_action != None:
                direction = self.past_action
            else:
                direction = np.random.randint(len(self.actions))
            direction = trans_aton(direction) / 2
            self.set_my_direction(int(direction))
        else:
            self.set_my_direction(self.next_direction)
        return self.my_direction

    def decide_next_direction(self, next_state):
        m = next_state
        if m.repr() in self.goalQ[self.my_goal][self.est_other_goal] and sum(self.goalQ[self.my_goal][self.est_other_goal][m.repr()]) != 0:
            direction = np.argmax(self.goalQ[self.my_goal][self.est_other_goal][m.repr()])
        elif self.past_action != None:
            direction = self.past_action
        else:
            direction = np.random.randint(len(self.actions))
        direction = trans_aton(direction) / 2
        self.next_direction = int(direction)
        return self.next_direction

    def get_rotate_direction(self, direction, angle): #return action_N
        #action rotate
        r_direction = direction + (angle / 90)
        #fix of action
        if r_direction >= 4:
            r_direction -= 4
        elif r_direction < 0:
            r_direction += 4

        return int(r_direction)

    def actplan_update(self, mode, relative_state=None, dir=None):
        if mode == "RESET":
            #TODO fix the method decide the actplan for the first time
            #とりあえずランダムにしているからどうしようかね．いかほどにしようかね
            r_row = relative_state.row
            r_col = relative_state.column
            if r_row <= 0:
                if r_col < 0:
                    go_right = 1
                    actplan = go_right
                elif r_col == 0:
                    go_right = 1
                    actplan = go_right
                elif r_col > 0:
                    go_left = 2
                    actplan = go_left
            else:
                straight = 0
                actplan = straight
            
            self.set_my_actplan(actplan)

        elif mode == "UPDATE":
            if self.actplan_one_change == False:
                #TODO fix the method update actplan process
                #update !!
                self.actplan_one_change = True
    def policy_update(self):
        if self.mode == self.GOAL_MODE:
            self.policy = self.goalQ[self.my_goal][self.est_other_goal]
        elif self.mode == self.ACTPLAN_MODE:
            self.policy = self.actplanQ[self.est_other_actplan]
    
    def get_my_direction(self, my_state=None):
        return self.my_direction

    def set_my_direction(self, my_direction):
        self.my_direction = my_direction
    def set_my_actplan(self, actplan):
        self.my_actplan = actplan
    
    