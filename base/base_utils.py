#適当に作った関数を入れとくファイル

from maze import Action
import json
from datetime import datetime as dt
import os

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