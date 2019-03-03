time_values = '''RHC,1,6,2764.0,66.0,97.668,10.493,0.046
RHC,151,6,2764.0,66.0,97.668,10.335,0.040
RHC,301,6,2764.0,66.0,97.668,10.283,0.111
RHC,451,6,2764.0,66.0,97.668,10.991,0.047'''


rows = time_values.split('\n')

import numpy as np
import matplotlib.pyplot as plt

algs = ['RHC', 'SA', 'GA']
iterations = []

iter_values = {}
for row in rows:
    values = row.split(',')
    alg = values[0]
    it_cnt = int(values[1])
    if it_cnt not in iterations:
        iterations.append(it_cnt)
    times = iter_values.get(alg)
    training_time = values[-2]
    if not times:
        times = [training_time]
        iter_values[alg] = times
    else:
        times.append(training_time)

fig = plt.figure()
plt.gca().set_color_cycle(['red', 'green', 'blue'])
keys = iter_values.keys()
for iter_value in keys:
    plt.plot(iterations, iter_values.get(iter_value))
plt.ylabel('Training Time')
plt.xlabel('Number of Restarts ')

plt.legend(algs, loc='upper left')

plt.title('Training Time vs Number of Restarts for RHC')
fig.savefig('Plots/Training Time vs Number of Restarts for RHC6.png')
plt.close()
