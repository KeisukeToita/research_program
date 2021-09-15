import random
from enum import Enum
import numpy as np
import math

from numpy.core.arrayprint import repr_format
from base_utils import *
from collections import defaultdict
from maze_8direction import *

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
        #reset goalQ
        self.goalQ = []
        for i in range(self.number_of_goals):
            self.goalQ.append([])
        for i in range(self.number_of_goals):
            for j in range(self.number_of_goals):
                self.goalQ[i].append(defaultdict(lambda: [0] * len(self.actions)))
        #reset actplanQ
        actplan_n = len(self.actplans)
        self.actplanQ = []
        for i in range(actplan_n):
            self.actplanQ.append([])
        for i in range(actplan_n):
            for j in range(actplan_n):
                self.actplanQ[i].append(defaultdict(lambda: [0] * len(self.actions)))

        self.policy_update()

    def episode_reset(self):
        self.est_other_goal = np.random.randint(self.number_of_goals)
        self.my_goal = np.random.randint(self.number_of_goals)

        self.goal_estimation_value = np.zeros(self.number_of_goals)
        self.goal_estimation_score = np.zeros(self.number_of_goals)

        self.goal_infer_steps = 0
        self.policy_update()

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
    def get_relative_state(self, my_state, other_state, o_Dir=None):
        m_r = my_state.row
        m_c = my_state.column
        o_c = other_state.column
        o_r = other_state.row
        #relative column & relative row
        r_r, r_c = o_r-m_r, o_c-m_c
        #direction
        if o_Dir == None:
            Dir = self.get_my_direction(my_state)
        else:
            Dir = o_Dir
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
    
    def estimate_other_actplan(self, other_state, other_action, my_state):
        other_action = trans_aton(other_action)
        if other_action != 8:
            #TODO infer part fix
            self.actplan_infer_steps += 1
            policy_each_actplan = []
            for other_actplan in range(len(self.actplans)):
                if not (0 in self.actplanQ[other_actplan][self.my_goal][o_r_s.repr()]):
                    policy_each_actplan.append(self.goalQ[other_actplan][self.my_goal][o_r_s.repr()][other_action]/ sum(self.goalQ[other_actplan][self.my_goal][o_r_s.repr()]))
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
        m_row = my_state.row
        m_col = my_state.column
        #mode judge
        mode = self.GOAL_MODE
        for x in range(-2, 3):
            for y in range(-2, 3):
                if other_state.repr() == State(m_row+y, m_col+x).repr():
                    mode = self.ACTPLAN_MODE
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
    def obs_other_past_action(self, action):
        self.other_past_action = action
    def set_mode(self, mode):
        self.mode = mode       