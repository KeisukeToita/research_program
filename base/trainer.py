#Training全体を受け持つ関数

class Trainer():

    def __init__(self, agent, env, episode):
        self.env = env
        self.agent = agent

        self.episode = episode

    

