time_values = '''SA,9,2,4503.0,98.0,97.870,6.362,0.054
SA,14,2,98.0,4503.0,2.130,6.690,0.069
SA,19,2,4503.0,98.0,97.870,6.856,0.052
SA,24,2,4503.0,98.0,97.870,6.246,0.052
SA,29,2,4503.0,98.0,97.870,6.641,0.050
SA,34,2,4503.0,98.0,97.870,6.334,0.054
SA,39,2,98.0,4503.0,2.130,6.321,0.077
SA,44,2,4503.0,98.0,97.870,8.059,0.052
SA,49,2,4503.0,98.0,97.870,6.683,0.053
SA,54,2,4503.0,98.0,97.870,6.336,0.045
SA,59,2,4503.0,98.0,97.870,6.256,0.055
SA,64,2,4503.0,98.0,97.870,6.141,0.060
SA,69,2,4503.0,98.0,97.870,6.217,0.044
SA,74,2,4503.0,98.0,97.870,6.255,0.051
SA,79,2,4503.0,98.0,97.870,6.192,0.054
SA,84,2,4503.0,98.0,97.870,7.818,0.058
SA,89,2,98.0,4503.0,2.130,7.440,0.054
SA,94,2,4503.0,98.0,97.870,6.266,0.055
SA,99,2,98.0,4503.0,2.130,6.184,0.055'''


rows = time_values.split('\n')

import numpy as np
import matplotlib.pyplot as plt

algs = ['SA']
iterations = []

iter_values = {}
for row in rows:
    values = row.split(',')
    alg = values[0]
    it_cnt = float(values[1])/100
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
plt.xlabel('Cooling Rate ')

plt.legend(algs, loc='upper left')

plt.title('Training Time vs Cooling Rate for SA')
fig.savefig('Plots/Training Time vs Cooling Rate for SA.png')
plt.close()
