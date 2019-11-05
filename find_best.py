import sys
import os
import pandas as pd
import numpy as np

nni_id = sys.argv[1]
#file_path = '~/nni/experiments/' + nni_id + '/log/RL_controller.log'
file_path = '/home/yujwang/nni/experiments/okBoeRjw/log/RL_controller.log'
os.environ['file_path'] = str(file_path)
os.system("grep 'reward' $file_path > tmp.txt")

data = pd.read_csv("tmp.txt", header=None, sep='\t')
data.columns = ['arc', 'arc_num', 'reward', 'reward_num']
max_reward = data.ix[data['reward_num'].idxmax()]
print("max_reward", max_reward)
arc_str = max_reward['arc_num']

fw = open("./scripts/arcs.sh", 'w')
fw.write("#!/bin/bash\n")

arc = [int(i) for i in arc_str.split(' ')]
start = 0
for layer_id in range(12):
    end = start + 1 + layer_id
    end += 1
    out_str = "fixed_arc=\"$fixed_arc {0}\"".format(np.reshape(arc[start: end], [-1]))
    out_str = out_str.replace("[", "").replace("]", "")
    print(out_str)
    fw.write(out_str+'\n')
    start = end

fw.close()