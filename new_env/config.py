import numpy as np
import pprint as PPAP

from new_env.maze import Multi_State
"""
各種実験設定用のクラスを集めているよ!!!!!!
__init__のところに設定の変数を追加していくのじゃ．
"""


"""
上位クラス conf
write：自分が持っている変数名と値を出力する関数
printのところを変えるとファイル出力もできるはず
"""
class conf:
    def __init__(self):
        pass
    def write(self):
        for key, value in self.__dict__.items():
           print("{} : {}".format(key, value))
"""
エリア(迷路)についての設定クラス
"""
class Area_conf(conf):
    
    def __init__(self):
        #縦と横の広さ
        self.WIDTH = 10
        self.HEIGHT = 10
        
        #4隅に配置してますよ．とりあえず．
        self.GOAL_DATA = [
            {"x_min":0,            "x_max":1,          "y_min":0,              "y_max":1 }, #GOAL 1
            {"x_min":self.WIDTH-1, "x_max":self.WIDTH, "y_min":0,              "y_max":1 }, #GOAL 2
            {"x_min":0,            "x_max":1,          "y_min":self.HEIGHT-1,  "y_max":self.HEIGHT}, # ...
            {"x_min":self.WIDTH-1, "x_max":self.WIDTH, "y_min":self.HEIGHT-1,  "y_max":self.HEIGHT }
        ]
        self.goal_num = len(self.GOAL_DATA)
    
    def goal_data(self, number):
        return self.GOAL_DATA[number]
    
    def _repr(self):
        print("~~~Maze Conf~~~")
        print("x : 0 ~ {}".format(self.WIDTH))
        print("y : 0 ~ {}".format(self.HEIGHT))
"""
行動についての設定クラス
"""
class Action_conf(conf):
    def __init__(self):
        self.ACTION_NUM = 3
        
        self.angle_range = 90
        
    def action_list_make(self):
        action_list = np.linspace(self.angle_range*-1, self.angle_range, self.ACTION_NUM)
        return action_list
"""
初期化用の設定クラス
"""   
class Reset_conf(conf):
    def __init__(self):
        self.ini_state_data = None
        self.ini_goal_data = None
"""
エージェントに関する設定クラス
"""
class Agent_conf(conf):
    def __init__(self):
        self.action_choice = "EPSILON" #"EPSILON","SOFTMAX"
        #Q_Learning conf
        self.epsilon = 0.3
        self.gamma = 0.9
        self.alpha = 0.1
        
        self.mode = "GOAL" #"GOAL","ACTPLAN"
        
        #Multi Agent conf
        self.value_type = "DISCRETE" #"DISCRETE","CONTINUOUS"
        self.agent_num = 2
"""
そのほか実験設定に関する設定クラス
""" 
class Experiment_conf(conf):
    def __init__(self):
        self.seed = 5
        self.episode = 100500
         
    