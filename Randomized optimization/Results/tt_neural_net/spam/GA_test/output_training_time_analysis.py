time_values = '''GA,10,1,1739,32,98.193,2764,66,97.668,47.492,0.028
GA,20,1,1739,32,98.193,2764,66,97.668,49.998,0.038
GA,30,1,1739,32,98.193,2764,66,97.668,53.898,0.031
GA,40,1,1739,32,98.193,2764,66,97.668,57.856,0.038'''


rows = time_values.split('\n')

import numpy as np
import matplotlib.pyplot as plt

algs = [ 'GA']
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
plt.xlabel('Mutation % ')

plt.legend(algs, loc='upper left')

plt.title('Training Time vs Mutation Size for GA')
fig.savefig('Plots/Training Time vs Mutation Percentage for GA.png')
plt.close()
