import numpy as np
import matplotlib.pyplot as plt

filename = 'errors_4_450.csv'
f = open(filename, 'r')
contents = f.read()
f.close()
splits = contents.split('---')

for i, split in enumerate(splits):
    print (i, split)
    if i == 1:
        rhc = split.split('\n')
    if i == 2:
        sa = split.split('\n')
    if i == 3:
        ga = split.split('\n')

n = []
err_rhc = []
err_sa = []
err_ga = []

for r, s, g in zip(rhc, sa, ga):
    r_values = r.split(',')
    s_values = s.split(',')
    g_values = g.split(',')

    if len(r_values) < 4:
        continue
    if len(s_values) < 4:
        continue
    if len(g_values) < 4:
        continue
    n.append(r_values[2])
    err_rhc.append(r_values[-1])
    err_sa.append(s_values[-1])
    err_ga.append(g_values[-1])

fig = plt.figure()

plt.gca().set_color_cycle(['red', 'green', 'blue'])
plt.plot(n, err_rhc)
plt.plot(n, err_sa)
plt.plot(n, err_ga)

plt.ylabel('Accuracy %')
plt.xlabel('Iterations')

algs = ['RHC', 'SA', 'GA']
plt.legend(algs, loc='lower right')
plt.title('Accuracy vs Iteration')

fig.savefig('Plots/Accuracy vs Iteration.png')
plt.close()
