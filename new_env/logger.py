def log_Q_table(Q__):
    for i, Q_ in enumerate(Q__):
        for j, Q in enumerate(Q_):
            with open("goal"+str(i)+"other_goal"+str(j)+"Q_table.txt", 'w', encoding='shift-jis', newline='') as f:
                for key in Q.keys():
                    f.write("{}:{}\n".format(key, Q[key]))
                f.close()

def log_estimate_rate(agent_num, rate):
    pass
