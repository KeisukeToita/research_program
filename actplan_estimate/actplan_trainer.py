from actplan_agent import *
from maze_8direction import *
from actplan_logger import *
from base_utils import *
import numpy as np
import copy


class Base_Trainer:
    def __init__(self, agents, env, episode=1, report_interval=50, write_interval=50, dirname=None, seed=1, is_colision=0):
        self.env = env
        self.agents = agents

        self.seed = seed
        self.episode = episode
        self.report_interval = report_interval
        self.write_interval = write_interval

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

        #show estimate rate

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
        if episode_n % self.write_interval == 0:
            self.logger.state_transition_write_csv(episode_n)
            for i in range(len(self.agents)):
                self.logger.q_table_write_csv(episode_n, self.agents[i].Q, i, self.env.row_length, self.env.column_length)
            
        return total_reward, step_n
    
class ActPlanTrainer(Base_Trainer):
    GOAL_FRAMEWORK = 0
    ACTPLAN_FRAMEWORK = 1
    def __init__(self, agents, env, episode=1, report_interval=50, write_interval=50, dirname=None, seed=1, is_colision=0):
        super().__init__(agents, env, episode=episode, report_interval=report_interval, write_interval=write_interval, dirname=dirname, seed=seed, is_colision=is_colision)

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
            self.agents[i].policy_update()
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
        if episode_n % self.write_interval == 0:
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

class ActPlanTrainer_with_direction(ActPlanTrainer):

    def __init__(self, agents, env, episode, report_interval, write_interval, dirname, seed, is_colision):
        super().__init__(agents, env, episode=episode, report_interval=report_interval, write_interval=write_interval,dirname=dirname, seed=seed, is_colision=is_colision)

    def one_seed(self, seed):
        print("{} seed train".format(seed+1))
        np.random.seed(seed)
        self.seed_reset()
        colision_count = self.one_seed_train(seed+1)

        goal_est_list = self.logger.estimate_goal_rate_ave
        actplan_est_list = self.logger.estimate_actplan_rate_ave
        reward_list = self.logger.total_reward_ave_log

        return [goal_est_list, actplan_est_list, reward_list, colision_count]

    def one_seed_train(self, seed):
        #train loop
        self.logger.seed_count(seed)
        colision_count = 0
        for i in range(self.episode):
            if (i+1) % self.report_interval == 0:
                reward, step = self.one_episode(i+1, seed)
                print("Episode {}: Agent gets {} reward. {} step".format(i+1, reward, step))
            else:
                reward, step = self.one_episode(i+1, seed)
            if reward < 0:
                colision_count += 1
        
        print("colision count:{}".format(colision_count))

        #estimate transition save
        self.logger.save_estimate_rate_move_ave("GOAL")
        # self.logger.save_estimate_rate_move_ave("ACTPLAN")

        return colision_count

    def one_episode(self, episode_n, seed):
        #init state & goal
        agents_state=[]
        agents_done=[]
        agents_goal=[]
        fix_goal=[2, 1]

        agents_state = self.env.reset(seed*episode_n)
        for i in range(len(self.agents)):
            agents_done.append(False)
            self.agents[i].episode_reset(fix_goal[i])
            # while self.agents[i].get_my_goal() in agents_goal:
            #     self.agents[i].episode_reset()
            agents_goal.append(self.agents[i].get_my_goal())

        other_goals=[self.agents[1].get_my_goal(), self.agents[0].get_my_goal()]

        self.env.set_agents_goal(agents_goal)

        total_reward = 0

        self.logger.init_list_of_dict()
        
        step_n = 1
        is_goal_step_exit = False

        #agent mode update & set frame part
        other_states = [agents_state[1], agents_state[0]]
        for i in range(len(self.agents)):
            self.agents[i].mode_update(agents_state[i], other_states[i])
            self.agents[i].policy_update()

        framework = self.agents[0].get_mode()
        #framework = self.GOAL_FRAMEWORK
        A_Frag = 0
        while False in agents_done:
            if framework == self.GOAL_FRAMEWORK:
                if A_Frag == 1:
                    A_Frag = 0
                    other_actplans = [self.agents[1].get_my_actplan(), self.agents[0].get_my_actplan()]
                    for i in range(len(self.agents)):
                        is_estimate = (other_actplans[i] == self.agents[i].get_est_other_actplan())
                        self.logger.add_is_estimate_actplan_one_episode(i, is_estimate)
                is_goal_step_exit = True
                next_states, agents_done, total_reward = self.goal_framework(step_n, agents_state, agents_done, total_reward)
            elif framework == self.ACTPLAN_FRAMEWORK:
                if A_Frag == 0:
                    for i in range(len(self.agents)):
                        self.agents[i].actplan_phase_reset()
                    first_flag = True
                    A_Frag = 1 
                else:
                    first_flag = False
                next_states, agents_done, total_reward = self.actplan_framework(step_n, agents_state, agents_done, total_reward, first_flag)

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

            

        #check the estimate quality
        if is_goal_step_exit:
            for i in range(len(self.agents)):
                is_estimate = (other_goals[i] == self.agents[i].get_est_other_goal())
                self.logger.add_is_estimate(i, is_estimate, "GOAL")
        else:
            for i in range(len(self.agents)):
                is_estimate = False
                self.logger.add_is_estimate(i, is_estimate, "GOAL")
        self.logger.add_estimate_rate_move_ave("GOAL")

        # if A_Frag == 1:
        #     other_actplans = [self.agents[1].get_my_actplan(), self.agents[0].get_my_actplan()]
        #     for i in range(len(self.agents)):
        #         is_estimate = (other_actplans[i] == self.agents[i].get_est_other_actplan())
        #         self.logger.add_is_estimate_actplan_one_episode(i, is_estimate)

        # for i in range(len(self.agents)):
        #     self.logger.add_is_estimate(i)
        # self.logger.add_estimate_rate_move_ave("ACTPLAN")
        
        self.logger.add_total_reward(total_reward)
        self.logger.add_total_reward_ave()
                
        #save with csv file
        if episode_n % self.write_interval == 0:
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

        #estimate
        other_states = [agents_state[1], agents_state[0]]
        other_actions = [actions[1], actions[0]]
        for i in range(len(self.agents)):
            self.agents[i].estimate_other_goal(other_states[i], other_actions[i])

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
        for i in range(len(self.agents)):
            self.agents[i].set_past_action(actions[i])
        return next_states, agents_done, total_reward

    def actplan_framework(self, step_n, agents_state, agents_done, total_reward, first_flag=False):
        actions, next_states, rewards=[],[],[]
            
        other_states = [agents_state[1], agents_state[0]]

        #make_relative_state & direction
        relative_states = []
        directions = []
        angles = []
        
        for i in range(len(self.agents)):
            directions.append(self.agents[i].decide_my_direction(agents_state[i]))
            r_state, angle = self.agents[i].get_relative_state(agents_state[i], other_states[i])
            relative_states.append(r_state)
            angles.append(angle)
        other_directions = [directions[1], directions[0]]

        #when the first actplan flamework step
        if first_flag == True:
            for i in range(len(self.agents)):
                self.agents[i].actplan_update("RESET", relative_states[i])

        #choice actions
        for i in range(len(self.agents)):
            actions.append(self.agents[i].act(relative_states[i], angles[i], other_directions[i]))

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
        next_directions = []
        next_angles = []
        for i in range(len(self.agents)):
            next_directions.append(self.agents[i].decide_next_direction(next_states[i]))
            r_state, angle = self.agents[i].get_relative_state(next_states[i], next_other_states[i], next_directions[i])
            next_relative_states.append(r_state)
            next_angles.append(angle)

        next_other_directions = [next_directions[1], next_directions[0]]

        for i in range(len(self.agents)):
            r_action = self.agents[i].get_rotate_action(trans_aton(actions[i]), angles[i]*-1)
            o_dir = self.agents[i].get_rotate_direction(other_directions[i], angles[i])
            n_o_dir = self.agents[i].get_rotate_direction(next_other_directions[i], next_angles[i])
            self.agents[i].learn(relative_states[i],
                                 next_relative_states[i],
                                 r_action,
                                 rewards[i],
                                 angles[i],
                                 o_dir,
                                 n_o_dir)

        #estimate
        
        other_relative_states = [relative_states[1], relative_states[0]]
        other_angles = [angles[1], angles[0]]
        other_actions = [actions[1], actions[0]]
        est_other_actplans=[]
        for i in range(len(self.agents)):
            r_other_action = self.agents[i].get_rotate_action(other_actions[i], other_angles[i]*-1)
            other_other_direction = self.agents[i].get_rotate_direction(self.agents[i].get_my_direction(), other_angles[i]*-1)
            est_other_actplans.append(self.agents[i].estimate_other_actplan(other_relative_states[i], r_other_action, other_other_direction))
        
    
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
                                   directions,
                                   other_directions,
                                   other_states,
                                   relative_states,
                                   my_goals,
                                   est_other_goals,
                                   my_actplans,
                                   est_other_actplans)

        #process to next step
        for i in range(len(self.agents)):
            #self.agents[i].actplan_update()
            self.agents[i].set_past_action(actions[i])

        return next_states, agents_done, total_reward