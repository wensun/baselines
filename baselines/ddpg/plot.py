##
# @file   plot.py
# @author Yibo Lin
# @date   Mar 2018
#

import matplotlib 
matplotlib.use('Agg')
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
        ret_all = None 
        ret_curr = None 
        for line in f:
            #if "eval:avg epi-ret (all)" in line: 
            #    match += 1
            search_epoch = re.search(r"\|\s+curr epoch id\s+\|\s+(\d+)\s+\|", line)
            if search_epoch: 
                epoch = int(search_epoch.group(1))
                #print("epoch ", epoch)
            search_ret_all = re.search(r"\|\s+eval:avg epi-ret \(all\)\s+\|\s+([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s+\|", line)
            if search_ret_all: 
                ret_all = float(search_ret_all.group(1))
            search_ret_curr = re.search(r"\|\s+eval:avg epi-ret \(curr\)\s+\|\s+([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s+\|", line)
            if search_ret_curr:
                ret_curr = float(search_ret_curr.group(1))
                assert epoch is not None
                assert ret_all is not None
                assert ret_curr is not None
                data.append([epoch, ret_all, ret_curr])
                if cumulative_flag and len(data) > 1: # use current best reward 
                    data[-1][1] = max(data[-1][1], data[-2][1])
                    data[-1][2] = max(data[-1][2], data[-2][2])
            count += 1
    return np.array(data)

seed_array = [29784, 15851, 12955, 1498, 21988, 7706, 31727, 2774, 8287, 20590]
alg_array = ["DDPG", "DDPGRM"]
#env = "InvertedPendulum-v2"
env = "InvertedDoublePendulum-v2"
env = "Ant-v2"
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

    p = plt.plot(avg_data[:, 0], avg_data[:, 1], label=alg)
    plt.fill_between(avg_data[:, 0], avg_data[:, 1]-std_data[:, 1], avg_data[:, 1]+std_data[:, 1], color=p[-1].get_color(), alpha=0.4)

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("cumulative ret_all" if cumulative_flag else "ret_all")
plt.title("%s" % env)
plt.tight_layout()
filename = "log/%s.%s.ret_all.png" % (env, alg)
print("save to %s" % (filename))
plt.savefig(filename, pad_inches=0.0)
filename = filename.replace("png", "pdf")
print("save to %s" % (filename))
plt.savefig(filename, pad_inches=0.0)

plt.clf()

# plot ret_curr 
for alg in alg_array: 
    data = data_map[alg]
    avg_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)

    print(avg_data)
    print(std_data)

    p = plt.plot(avg_data[:, 0], avg_data[:, 2], label=alg)
    plt.fill_between(avg_data[:, 0], avg_data[:, 2]-std_data[:, 2], avg_data[:, 2]+std_data[:, 2], color=p[-1].get_color(), alpha=0.4)

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("cumulative ret_curr" if cumulative_flag else "ret_curr")
plt.title("%s" % env)
plt.tight_layout()
filename = "log/%s.%s.ret_curr.png" % (env, alg)
print("save to %s" % (filename))
plt.savefig(filename, pad_inches=0.0)
filename = filename.replace("png", "pdf")
print("save to %s" % (filename))
plt.savefig(filename, pad_inches=0.0)
