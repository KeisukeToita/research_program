from actplan_agent import *
from maze_8direction import *
from actplan_logger import *
from actplan_trainer import *
from base_utils import *
import shutil

def main():
    #実験設定の読み込み
    config_dir = "./config/"
    config = load_json(config_dir+"experiment_edit.txt")

    #結果を格納するディレクトリを準備
    dirname = resultdir_make(config["exp_title"])
    shutil.copyfile(config_dir+"experiment_edit.txt", dirname+"config.txt")

    #環境データ
    grid = maze_open(config_dir+config["maze_data"])
    env = Maze_8direction(grid, get_init_agents_state(config), agent_num=config["agent_num"], is_goal=config["is_goal"])

    #エージェント
    agents=[]
    for i in range(config["agent_num"]):
        #agents.append(Base_Agent(env, epsilon=config["epsilon"], gamma=config["gamma"], alpha=config["alpha"]))
        agents.append(ActPlanAgent(env, config["agent_goal_num"], epsilon=config["epsilon"], gamma=config["gamma"], alpha=config["alpha"], act_mode=config["act_mode"]))
        
    #トレーナー
    #trainer = SOMQLearningTrainer_Gchange(agents, env, config["episode"], config["repo&write_interval"], dirname, config['seed'])
    """
    trainer = Base_Trainer(agents = agents,
                            env = env,
                            episode = config["episode"],
                            report_interval = config["repo&write_interval"],
                            dirname = dirname,
                            seed = config['seed'],
                            is_colision = config["is_colision"])
    """
    trainer = ActPlanTrainer(agents = agents,
                            env = env,
                            episode = config["episode"],
                            report_interval = config["repo&write_interval"],
                            dirname = dirname,
                            seed = config['seed'],
                            is_colision = config["is_colision"])

    trainer.all_seed_train()


if __name__ == "__main__":
    main()
