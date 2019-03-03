time_values = '''GA,10,5,2764.0,66.0,97.668,53.395,0.045
GA,30,5,2764.0,66.0,97.668,62.469,0.044
GA,50,5,2764.0,66.0,97.668,61.576,0.041
GA,70,5,2764.0,66.0,97.668,67.882,0.057
GA,90,5,2764.0,66.0,97.668,67.791,0.057'''


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
plt.xlabel('Population Percentage ( % ) ')

plt.legend(algs, loc='upper left')

plt.title('Training Time vs Population Percentage for GA')
fig.savefig('Plots/Training Time vs Population Percentage for GA.png')
plt.close()
