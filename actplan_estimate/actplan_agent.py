import random
from enum import Enum
import numpy as np
import math
from base_utils import *
from collections import defaultdict

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

class ActplanAgent(Base_Agent):
    GOAL_MODE = 0
    ACTPLAN_MODE = 1

    def __init__(self, env, number_of_goals, epsilon=0.1, gamma=0.9, alpha=0.1):
        super().__init__(env, epsilon, gamma, alpha)

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

        """ infer_part
        self.goal_infer_steps = 0
        self.goal_estimation_value = np.zeros(number_of_goals)
        self.goal_estimation_score = np.zeros(number_of_goals)
        """
        
        #about actplan
        self.number_of_actplans = 4
        self.actplanQ = []
        for i in range(self.number_of_actplans):
            self.actplanQ.append([])
        for i in range(self.number_of_actplans):
            for j in range(self.number_of_actplans):
                self.actplanQ[i].append(defaultdict(lambda: [0] * len(self.actions)))

        self.my_actplan = np.random.randint(self.number_of_actplans)
        self.est_other_actplan = np.random.randint(self.number_of_actplans)

        """ infer_part
        self.actplan_infer_steps = 0
        self.actplan_estimation_value = np.zeros(self.number_of_actions)
        self.actplan_estimation_score = np.zeros(sefl.number_of_actions)
        """
        
        self.policy = self.Q[self.my_goal][self.est_other_goal]

    #get func
    def get_est_other_goal(self):
        pass

    def get_goal_estimation_value(self):
        pass

    def get_est_other_actplan(self):
        pass

    def get_actplan_estimation_value(self):
        pass

       