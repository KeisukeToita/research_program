from agent import *
from maze import *
from logger import *
from trainer import *
from base_utils import *

def main():
    #実験設定の読み込み
    config = load_json("experiment_edit.txt")

    #結果を格納するディレクトリを準備
    dirname = resultdir_make(config["exp_title"])

    #環境データ
    grid = maze_open(config["maze_data"])
    env = Maze(grid,agent_num=config["agent_num"])

    #エージェント
    agents=[]
    #agent = QLearningAgent(env, epsilon=0.3)
    for i in range(config["agent_num"]):
        agents.append(Agent(env))
    

    #trainer = QLearningTrainer(agent, env, config["episode"], config["repo&write_interval"], dirname)
    trainer = Trainer(agents, env, config["episode"], config["repo&write_interval"], dirname)

    trainer.train()


if __name__ == "__main__":
    main()
