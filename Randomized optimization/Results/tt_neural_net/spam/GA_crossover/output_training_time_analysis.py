time_values = '''GA,10,1,2764.0,66.0,97.668,18.746,0.037
GA,30,1,2764.0,66.0,97.668,35.602,0.049
GA,50,1,2764.0,66.0,97.668,51.812,0.043
GA,70,1,2764.0,66.0,97.668,61.193,0.057
GA,90,1,2764.0,66.0,97.668,69.111,0.061'''


rows = time_values.split('\n')

import numpy as np
import matplotlib.pyplot as plt

algs = ['GA']
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
plt.xlabel('Crossover Probability ( % )')

plt.legend(algs, loc='upper left')

plt.title('Training Time vs Crossover Probability for GA')
fig.savefig('Plots/Training Time vs Crossover Probability for GA.png')
plt.close()
