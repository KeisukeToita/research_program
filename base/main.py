from agent import *
from maze import *
from logger import *
from trainer import *
from base_utils import *
import shutil

def main():
    #実験設定の読み込み
    config_dir = "./config/"
    config = load_json(config_dir+"SOMexperiment_edit.txt")

    #結果を格納するディレクトリを準備
    dirname = resultdir_make(config["exp_title"])
    shutil.copyfile(config_dir+"SOMexperiment_edit.txt", dirname+"config.txt")

    #環境データ
    grid = maze_open(config_dir+config["maze_data"])
    env = Maze(grid, get_init_agents_state(config), agent_num=config["agent_num"], is_goal=config["is_goal"])

    #エージェント
    agents=[]
    for i in range(config["agent_num"]):
        #agents.append(QLearningAgent(env ,epsilon=0.2))
        #agents.append(SOMQLearningAgent(env, number_of_goals=config["agent_goal_num"], epsilon=config["epsilon"], gamma=config["gamma"], alpha=config["alpha"]))
        agents.append(SOMQLearningAgent_Gchange(env, number_of_goals=config["agent_goal_num"], epsilon=config["epsilon"], gamma=config["gamma"], alpha=config["alpha"], g_change_rate=config["goal_change_rate"]))
        
    #トレーナー
    #trainer = QLearningTrainer(agents, env, config["episode"], config["repo&write_interval"], dirname)
    #trainer = Trainer(agents, env, config["episode"], config["repo&write_interval"], dirname)
    trainer = SOMQLearningTrainer_Gchange(agents, env, config["episode"], config["repo&write_interval"], dirname, config['seed'])
    #trainer = SOMQLearningTrainer(agents, env, config["episode"], config["repo&write_interval"], dirname, config['seed'])

    trainer.all_seed_train()


if __name__ == "__main__":
    main()
