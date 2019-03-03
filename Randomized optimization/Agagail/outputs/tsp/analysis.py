f = open('iteration_time_analysis.csv', 'r')
contents = f.read()
f.close()

n = []
rhc = []
sa = []
ga = []
mimic = []

rows = contents.split('\n')
for i, row in enumerate(rows):
    values = row.split(',')
    n.append(values[0])
    rhc.append(values[1])
    sa.append(values[2])
    ga.append(values[3])
    mimic.append(values[4])


import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
plt.gca().set_color_cycle(['red', 'green', 'blue', 'orange'])

plt.plot(n, rhc)
plt.plot(n, sa)
plt.plot(n, ga)
# plt.plot(n, mimic)

plt.ylabel('Time')
plt.xlabel('Number of Cities')

algs = ['RHC', 'SA', 'GA']#, 'MIMIC']
plt.legend(algs, loc='upper left')

plt.title('Number of Cities vs Time')
fig.savefig('Plots/Number of Cities vs Time (no mimic).png')
plt.close()
