#適当に作った関数を入れとくファイル

from maze_8direction import *
import json
from datetime import datetime as dt
import os
import sys

#function about agent
def trans_ntoa(action_n):
    if action_n == 0:
        return Action.U
    if action_n == 1:
        return Action.UR
    if action_n == 2:
        return Action.R
    if action_n == 3:
        return Action.DR
    if action_n == 4:
        return Action.D
    if action_n == 5:
        return Action.DL
    if action_n == 6:
        return Action.L
    if action_n == 7:
        return Action.UL
    if action_n == 8:
        return Action.S

def trans_aton(action):
    if action == Action.U:
        return 0
    if action == Action.UR:
        return 1
    if action == Action.R:
        return 2
    if action == Action.DR:
        return 3
    if action == Action.D:
        return 4
    if action == Action.DL:
        return 5
    if action == Action.L:
        return 6
    if action == Action.UL:
        return 7
    if action == Action.S:
        return 8

#function about data loading
def load_json(filename):
    f = open(filename, 'r')
    text_data = f.readlines()
    experiment_data = ''
    for data in text_data:
        experiment_data += data
    experiment_data = json.loads(experiment_data)
    return experiment_data

#function about making directry
def resultdir_make(title):
    tdatetime = dt.now()
    tstr = tdatetime.strftime('%Y%m%d%H%M')
    dirname = tstr + "_" + title
    predir = "../../result/"
    newdir = predir+dirname+"/"
    os.mkdir(newdir)
    return newdir

def maze_open(file):
    data = []
    try:
        f = open(file, 'r', encoding='utf-8')
    except Exception:
        print("open error. not found file:", str(file))
        sys.exit(1)
    for line in f:
        line = line.strip() #前後空白削除
        line = line.replace('\n','') #末尾の\nの削除
        line = line.split(" ") #分割
        for i in range(len(line)):
            line[i] = int(line[i])
        data.append(line)
    f.close()
    return data

def get_init_agents_state(config):
    init_agents_state=[]
    for i in range(config["agent_num"]):
        s=config["init_agent_state_"+str(i+1)]
        state=State(s["row"], s["column"])
        init_agents_state.append(state)

    return init_agents_state
