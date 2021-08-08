#適当に作った関数を入れとくファイル

from maze import Action
import json
from datetime import datetime as dt
import os
import sys

#function about agent
def trans_ntoa(action_n):
    if action_n == 0:
        return Action.UP
    if action_n == 1:
        return Action.DOWN
    if action_n == 2:
        return Action.LEFT
    if action_n == 3:
        return Action.RIGHT
    if action_n == 4:
        return Action.STAY

def trans_aton(action):
    if action == Action.UP:
        return 0
    if action == Action.DOWN:
        return 1
    if action == Action.LEFT:
        return 2
    if action == Action.RIGHT:
        return 3
    if action == Action.STAY:
        return 4

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
    predir = "../result/"
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