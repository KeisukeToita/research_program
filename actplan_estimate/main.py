from actplan_agent import *
from maze_8direction import *
from actplan_logger import *
from actplan_trainer import *
from base_utils import *
import shutil
from joblib import Parallel, delayed

def prepare():
    #実験設定の読み込み
    config_dir = "./config/"
    config = load_json(config_dir+"experiment_edit.txt")

    #結果を格納するディレクトリを準備
    dirname = resultdir_make(config["exp_title"])
    shutil.copyfile(config_dir+"experiment_edit.txt", dirname+"config.txt")

    return config, dirname

def exp(seed, config, dirname):
    config_dir = "./config/"
    #環境データ
    grid = maze_open(config_dir+config["maze_data"])
    env = Maze_8direction(grid, agent_num=config["agent_num"], init_agents_state=get_init_agents_state(config), is_goal=config["is_goal"])
    
    #エージェント
    agents=[]
    for i in range(config["agent_num"]):
        #agents.append(Base_Agent(env, epsilon=config["epsilon"], gamma=config["gamma"], alpha=config["alpha"]))
        #agents.append(ActPlanAgent(env, config["agent_goal_num"], epsilon=config["epsilon"], gamma=config["gamma"], alpha=config["alpha"], act_mode=config["act_mode"]))
        agents.append(ActPlanAgent_with_direction(env, config["agent_goal_num"], epsilon=config["epsilon"], gamma=config["gamma"], alpha=config["alpha"], act_mode=config["act_mode"]))
    
    """
    TODO エージェントにロードしたQテーブルを読み込ませる．
    """
    log_dir_path = "../../result/202109231449_important_data_Actplan_agent_3direction/"

    loader = Data_loader(log_dir_path, config["agent_num"])

    for i in range(config["agent_num"]):
        goalQ = loader.goalQ_table_load(i)
        actplanQ = loader.actplanQ_table_load(i)
        agents[i].goalQ = goalQ
        agents[i].actplanQ = actplanQ

    #トレーナー
    #trainer = SOMQLearningTrainer_Gchange(agents, env, config["episode"], config["repo&write_interval"], dirname, config['seed'])
    """
    trainer = Base_Trainer(agents = agents,
                            env = env,
                            episode = config["episode"],
                            report_interval = config["repo&write_interval"],
                            write_interval=config["write_interval"],
                            dirname = dirname,
                            seed = config['seed'],
                            is_colision = config["is_colision"])
    """
    """
    trainer = ActPlanTrainer(agents = agents,
                            env = env,
                            episode = config["episode"],
                            report_interval = config["repo&write_interval"],
                            write_interval=config["write_interval"],
                            dirname = dirname,
                            seed = config['seed'],
                            is_colision = config["is_colision"])
    """
    
    trainer = ActPlanTrainer_with_direction(agents = agents,
                                            env = env,
                                            episode = config["episode"],
                                            report_interval = config["report_interval"],
                                            write_interval=config["write_interval"],
                                            dirname = dirname,
                                            seed = config['seed'],
                                            is_colision = config["is_colision"])
    data = trainer.one_seed(seed)
    return data
    
def main():
    config, dirname = prepare()
    all_logger = AllSeedLogger(dirname, config["agent_num"])
    v = Parallel(n_jobs=-1)([delayed(exp)(i, config, dirname) for i in range(config["seed"])] )

    #make the all seed data
    all_logger.add_one_seed_data(v)
    all_logger.make_data()

    all_logger.save_all_estimate_rate_move_ave("GOAL")
    # all_logger.save_all_estimate_rate_move_ave("ACTPLAN")

    all_logger.save_all_total_reward_move_ave()
    all_logger.write_colision_count_ave()

if __name__ == "__main__":
    main()
