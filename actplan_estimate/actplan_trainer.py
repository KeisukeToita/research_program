from actplan_agent import *
from maze_8direction import *
from actplan_logger import *
from base_utils import *
import numpy as np
import copy

np.random.seed(0)

class Base_Trainer:
    def __init__(self, agents, env, episode=1, report_interval=50, dirname=None, seed=1, is_colision=0):
        self.env = env
        self.agents = agents

        self.seed = seed
        self.episode = episode
        self.report_interval = report_interval

        self.is_colision = is_colision

        self.logger = Actplan_Logger(dirname, len(agents))

    def seed_reset(self):
        for i in range(len(self.agents)):
            self.agents[i].seed_reset()

    def all_seed_train(self):
        print("~~~ Learning_START ~~~")
        for i in range(self.seed):
            print("{} seed train".format(i+1))
            self.seed_reset()
            self.one_seed_train()        

    def one_seed_train(self):
        #train loop
        self.logger.seed_count()
        colision_count = 0
        for i in range(self.episode):
            if (i+1) % self.report_interval == 0:
                reward, step = self.one_episode(i+1)
                print("Episode {}: Agent gets {} reward. {} step".format(i+1, reward, step))
            else:
                reward, step = self.one_episode(i+1)
            if reward < 0:
                colision_count += 1
        
        print("colision count:{}".format(colision_count))

        #estimate transition save
        """
        self.logger.show_estimate_rate()
        """

        #show estimate rate
        """
        for i in range(len(self.agents)):
            self.estimate_rates[i] = self.estimate_counts[i]/self.episode
            print("Agent {} rate: {}".format(i, self.estimate_rates[i]))
        """

    def one_episode(self, episode_n):
        #init state & goal
        agents_state=[]
        agents_done=[]

        agents_state = self.env.reset()
        for i in range(len(self.agents)):
            agents_done.append(False)
            self.agents[i].episode_reset()

        total_reward = 0

        self.logger.init_list_of_dict()
        
        step_n = 1

        while False in agents_done:
            actions, next_states, rewards=[],[],[]
            
            #choice actions
            for i in range(len(self.agents)):
                actions.append(self.agents[i].greedy_act(agents_state[i]))
            
            #one step
            for i in range(len(self.agents)):
                if agents_done[i] == False:
                    next_state, reward, done = self.env.step(i,actions[i])
                else:
                    actions[i]=self.env.actions[8] #STAY
                    next_state, reward, done = self.env.get_finish_states(i), int(0), True

                next_states.append(next_state)
                rewards.append(reward)
                agents_done[i]=done
            
            #colision_judge part
            if self.is_colision == 1:
                rewards, agents_done = self.env.colision_judge(next_states, rewards, agents_done)

            #learn
            for i in range(len(self.agents)):
                self.agents[i].learn(agents_state[i], next_states[i], actions[i], rewards[i])
           
            #calcurate total reward
            total_reward += sum(rewards)
            self.logger.add_experience(step_n, agents_state, actions, rewards, total_reward)
           
            #move next_state
            for i in range(len(self.agents)):
                agents_state[i] = next_states[i]
           
            step_n += 1
                
        #save with csv file
        if episode_n % self.report_interval == 0:
            self.logger.state_transition_write_csv(episode_n)
            for i in range(len(self.agents)):
                self.logger.q_table_write_csv(episode_n, self.agents[i].Q, i, self.env.row_length, self.env.column_length)
            
        return total_reward, step_n
    

class ActPlanTrainer(Base_Trainer):
    GOAL_FRAMEWORK = 0
    ACTPLAN_FRAMEWORK = 1
    def __init__(self, agents, env, episode=1, report_interval=50, dirname=None, seed=1, is_colision=0):
        super().__init__(agents, env, episode, report_interval, dirname, seed, is_colision)

    def one_episode(self, episode_n):
        #init state & goal
        agents_state=[]
        agents_done=[]
        agents_goal=[]

        agents_state = self.env.reset()
        for i in range(len(self.agents)):
            agents_done.append(False)
            self.agents[i].episode_reset()
            while self.agents[i].get_my_goal() in agents_goal:
                self.agents[i].episode_reset()
            agents_goal.append(self.agents[i].get_my_goal())

        other_goals=[self.agents[1].get_my_goal(), self.agents[0].get_my_goal()]

        self.env.set_agents_goal(agents_goal)

        total_reward = 0

        self.logger.init_list_of_dict()
        
        step_n = 1

        #TODO agent mode update & set frame part
        other_states = [agents_state[1], agents_state[0]]
        for i in range(len(self.agents)):
            self.agents[i].mode_update(agents_state[i], other_states[i])
        framework = self.agents[0].get_mode()
        A_Frag = 0
        while False in agents_done:
            if framework == self.GOAL_FRAMEWORK:
                A_Frag = 0
                next_states, agents_done, total_reward = self.goal_framework(step_n, agents_state, agents_done, total_reward)
            elif framework == self.ACTPLAN_FRAMEWORK:
                if A_Frag == 0:
                    for i in range(len(self.agents)):
                        self.agents[i].actplan_phase_reset()
                    A_Frag = 1 
                next_states, agents_done, total_reward = self.actplan_framework(step_n, agents_state, agents_done, total_reward)
            #move next_state
            for i in range(len(self.agents)):
                agents_state[i] = next_states[i]
            
            #mode update
            n_other_states = [agents_state[1], agents_state[0]]
            for i in range(len(self.agents)):
                self.agents[i].mode_update(agents_state[i], n_other_states[i])
            framework = self.agents[0].get_mode()

            #policy update
            for i in range(len(self.agents)):
                self.agents[i].policy_update()
            step_n += 1
            
                
        #save with csv file
        if episode_n % self.report_interval == 0:
            self.logger.state_transition_write_csv(episode_n)
            for i in range(len(self.agents)):
                self.logger.all_q_table_write_csv(episode_n, self.agents[i].goalQ, i, self.env.row_length, self.env.column_length, "GOAL")
                self.logger.all_q_table_write_csv(episode_n, self.agents[i].actplanQ, i, self.env.row_length, self.env.column_length, "ACTPLAN")
        return total_reward, step_n

    def goal_framework(self, step_n, agents_state, agents_done, total_reward):
        actions, next_states, rewards=[],[],[]           
        #choice actions
        for i in range(len(self.agents)):
            actions.append(self.agents[i].act(agents_state[i]))
        #one step
        for i in range(len(self.agents)):
            if agents_done[i] == False:
                next_state, reward, done = self.env.step(i,actions[i])
            else:
                actions[i]=self.env.actions[8] #STAY
                next_state, reward, done = self.env.get_finish_states(i), int(0), True
            next_states.append(next_state)
            rewards.append(reward)
            agents_done[i]=done
        #colision_judge part
        if self.is_colision == 1:
            rewards, agents_done = self.env.colision_judge(agents_state, next_states, rewards, agents_done)
        #learn
        for i in range(len(self.agents)):
            self.agents[i].learn(agents_state[i], next_states[i], actions[i], rewards[i])
        #TODO estimate
        """
        skip now
        """
        #calcurate total reward
        total_reward += sum(rewards)
        #log experience
        my_goals = []
        est_other_goals = []
        for i in range(len(self.agents)):
            my_goals.append(self.agents[i].get_my_goal())
            est_other_goals.append(self.agents[i].get_est_other_goal())
        self.logger.add_experience(step_n,
                                   agents_state,
                                   actions,
                                   rewards,
                                   total_reward,
                                   "with ACTPLAN",
                                   "GOAL",
                                   my_goals=my_goals,
                                   est_other_goals=est_other_goals)
        #process to next step
        """
        set my past action and other past action
        """
        other_action = [actions[1], actions[0]]
        for i in range(len(self.agents)):
            self.agents[i].set_past_action(actions[i])
            self.agents[i].obs_other_past_action(other_action[i])
        return next_states, agents_done, total_reward

    def actplan_framework(self, step_n, agents_state, agents_done, total_reward): #2 agent only
        actions, next_states, rewards=[],[],[]
            
        other_states = [agents_state[1], agents_state[0]]

        #make_relative_state
        relative_states = []
        angles = []
        for i in range(len(self.agents)):
            r_state, angle = self.agents[i].get_relative_state(agents_state[i], other_states[i])
            relative_states.append(r_state)
            angles.append(angle)
        #choice actions
        for i in range(len(self.agents)):
            actions.append(self.agents[i].act(relative_states[i], angles[i]))
        
        #one step
        for i in range(len(self.agents)):
            if agents_done[i] == False:
                next_state, reward, done = self.env.step(i,actions[i])
            else:
                actions[i]=self.env.actions[8] #STAY
                next_state, reward, done = self.env.get_finish_states(i), int(0), True

            next_states.append(next_state)
            rewards.append(reward)
            agents_done[i]=done
        
        #colision_judge part
        if self.is_colision == 1:
            rewards, agents_done = self.env.colision_judge(agents_state, next_states, rewards, agents_done)
        #learn
        next_other_states = [next_states[1], next_states[0]]
        next_relative_states = []
        next_angles = []
        for i in range(len(self.agents)):
            r_state, angle = self.agents[i].get_relative_state(next_states[i], next_other_states[i])
            next_relative_states.append(r_state)
            next_angles.append(angle)

        for i in range(len(self.agents)):
            r_action = self.agents[i].get_rotate_action(trans_aton(actions[i]), angles[i]*-1)
            self.agents[i].learn(relative_states[i], next_relative_states[i], r_action, rewards[i], angles[i])

        #TODO estimate
        """
        skip now
        """
        #calcurate total reward
        total_reward += sum(rewards)
        #log experience
        my_goals, est_other_goals, my_actplans, est_other_actplans = [], [], [], []
        for i in range(len(self.agents)):
            my_goals.append(self.agents[i].get_my_goal())
            est_other_goals.append(self.agents[i].get_est_other_goal())
            my_actplans.append(self.agents[i].get_my_actplan())
            est_other_actplans.append(self.agents[i].get_est_other_actplan())
        self.logger.add_experience(step_n,
                                   agents_state,
                                   actions,
                                   rewards,
                                   total_reward,
                                   "with ACTPLAN",
                                   "ACTPLAN",
                                   other_states,
                                   relative_states,
                                   my_goals,
                                   est_other_goals,
                                   my_actplans,
                                   est_other_actplans)

        #process to next step
        other_action = [actions[1], actions[0]]
        for i in range(len(self.agents)):
            self.agents[i].set_past_action(actions[i])
            self.agents[i].obs_other_past_action(other_action[i])

        return next_states, agents_done, total_reward