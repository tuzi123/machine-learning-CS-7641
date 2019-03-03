import numpy as np
import matplotlib.pyplot as plt

filename = 'errors_train_test_5_650.csv'
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
    err_rhc.append(r_values[3])
    err_sa.append(s_values[3])
    err_ga.append(g_values[3])

fig = plt.figure()

plt.gca().set_color_cycle(['red', 'green', 'blue'])
plt.plot(n, err_rhc)
plt.plot(n, err_sa)
plt.plot(n, err_ga)

plt.ylabel('Error')
plt.xlabel('Iterations')

algs = ['RHC', 'SA', 'GA']
plt.legend(algs, loc='upper right')
plt.title('Error vs Iteration')

fig.savefig('Plots/Error vs Iteration.png')
plt.close()
