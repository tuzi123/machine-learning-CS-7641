import csv


f = open('optimal_value_analysis_100.csv', 'r')
contents = f.read()
f.close()
# print (values)

rows = contents.split('\n')

import numpy as np
import matplotlib.pyplot as plt

x = []
rhc = []
sa = []
ga = []
mimic = []

for row in rows:
    values = row.split(',')
    x.append(values[0])
    rhc.append(values[1])
    sa.append(values[2])
    ga.append(values[3])
    mimic.append(values[4])

fig = plt.figure()
plt.gca().set_color_cycle(['red', 'green', 'blue', 'orange'])
plt.plot(x, rhc)
plt.plot(x, sa)
plt.plot(x, ga)
plt.plot(x, mimic)
plt.ylabel('Optimal Value')
plt.xlabel('N')

plt.legend(['RHC', 'SA', 'GA', 'MIMIC'], loc='upper left')

plt.title('Optimal Value vs N')
fig.savefig('Plots/Distance vs N.png')
plt.close()
