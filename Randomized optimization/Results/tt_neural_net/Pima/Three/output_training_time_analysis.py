time_values = '''RHC,50,2,657.0,111.0,85.547,0.612,0.024
SA,50,2,657.0,111.0,85.547,0.276,0.021
GA,50,2,657.0,111.0,85.547,2.193,0.009
RHC,150,2,657.0,111.0,85.547,1.441,0.093
SA,150,2,111.0,657.0,14.453,0.879,0.006
GA,150,2,657.0,111.0,85.547,6.326,0.005
RHC,250,2,657.0,111.0,85.547,1.815,0.035
SA,250,2,657.0,111.0,85.547,1.129,0.008
GA,250,2,657.0,111.0,85.547,13.527,0.019
RHC,350,2,657.0,111.0,85.547,3.329,0.048
SA,350,2,657.0,111.0,85.547,2.058,0.007
GA,350,2,657.0,111.0,85.547,16.497,0.006
RHC,450,2,657.0,111.0,85.547,4.808,0.041
SA,450,2,657.0,111.0,85.547,2.570,0.007
GA,450,2,657.0,111.0,85.547,22.311,0.008
RHC,550,2,657.0,111.0,85.547,4.810,0.026
SA,550,2,657.0,111.0,85.547,2.125,0.005
GA,550,2,657.0,111.0,85.547,23.243,0.005
RHC,650,2,657.0,111.0,85.547,3.672,0.020
SA,650,2,657.0,111.0,85.547,2.118,0.005
GA,650,2,657.0,111.0,85.547,26.517,0.008'''


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
