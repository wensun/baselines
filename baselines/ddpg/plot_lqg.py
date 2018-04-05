##
# @file   plot_lqg.py
# @author Yibo Lin
# @date   Apr 2018
#

import matplotlib.pyplot as plt 
import numpy as np 
import re 
import glob 

def read(filename, cumulative_flag):
    print("reading %s" % (filename))
    data = []
    with open(filename, "r") as f: 
        count = 1 
        epoch = None 
        cost = None 
        for line in f:
            search_cost = re.search(r"at epoch (\d+), the current policy has cost ([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)", line)
            if search_cost: 
                epoch = float(search_cost.group(1))
                cost = float(search_cost.group(2))
                assert epoch is not None
                assert cost is not None
                data.append([epoch, cost])
                if cumulative_flag and len(data) > 1: # use current best reward 
                    data[-1][1] = min(data[-1][1], data[-2][1])
            count += 1
    return np.array(data)

seed_array = [29784, 15851, 12955, 1498, 21988, 7706, 31727, 2774, 8287, 20590]
alg_array = ["Plain_GD", "Natural_GD", "Newton_Natural_GD"]
env = "LQG"
cumulative_flag = True

data_map = {}
for alg in alg_array:
    data = []
    for seed in seed_array:
        entry = read("log/%s.%s.seed%d.log" % (env, alg, seed), cumulative_flag)
        print(entry.shape)
        data.append(entry)
    data = np.array(data)
    data_map[alg] = data

# plot ret_all 
for alg in alg_array: 
    data = data_map[alg]
    avg_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)

    print(avg_data)
    print(std_data)

    p = plt.semilogy(avg_data[:, 0], avg_data[:, 1], label=alg)
    plt.fill_between(avg_data[:, 0], avg_data[:, 1]-std_data[:, 1], avg_data[:, 1]+std_data[:, 1], where=avg_data[:, 1]-std_data[:, 1] > 0, color=p[-1].get_color(), alpha=0.4)

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("cumulative cost" if cumulative_flag else "cost")
plt.title("%s" % env)
plt.tight_layout()
filename = "log/%s.%s.cost.png" % (env, alg)
print("save to %s" % (filename))
plt.savefig(filename, pad_inches=0.0)
filename = filename.replace("png", "pdf")
print("save to %s" % (filename))
plt.savefig(filename, pad_inches=0.0)
