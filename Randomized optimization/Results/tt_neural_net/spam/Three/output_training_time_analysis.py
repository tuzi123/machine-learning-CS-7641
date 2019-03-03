time_values = '''RHC,50,5,98.0,4503.0,2.130,1.510,0.050
SA,50,5,98.0,4503.0,2.130,1.019,0.024
GA,50,5,4503.0,98.0,97.870,10.849,0.020
RHC,150,5,98.0,4503.0,2.130,3.709,0.063
SA,150,5,98.0,4503.0,2.130,1.829,0.016
GA,150,5,4503.0,98.0,97.870,30.197,0.018
RHC,250,5,4503.0,98.0,97.870,5.334,0.047
SA,250,5,4503.0,98.0,97.870,3.082,0.018
GA,250,5,4503.0,98.0,97.870,50.011,0.024
RHC,350,5,4503.0,98.0,97.870,7.255,0.070
SA,350,5,98.0,4503.0,2.130,5.116,0.017
GA,350,5,4503.0,98.0,97.870,72.549,0.024
RHC,450,5,4503.0,98.0,97.870,9.530,0.062
SA,450,5,4503.0,98.0,97.870,6.656,0.024
GA,450,5,4503.0,98.0,97.870,91.765,0.046
RHC,550,5,4503.0,98.0,97.870,11.072,0.060
SA,550,5,4503.0,98.0,97.870,6.559,0.021
GA,550,5,4503.0,98.0,97.870,117.373,0.019
RHC,650,5,4503.0,98.0,97.870,13.381,0.055
SA,650,5,4503.0,98.0,97.870,9.223,0.018
GA,650,5,4503.0,98.0,97.870,135.475,0.017'''


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
plt.xlabel('Iterations')

plt.legend(algs, loc='upper left')

plt.title('Training Time vs Iteration')
fig.savefig('Plots/Training Time vs Iteration.png')
plt.close()
