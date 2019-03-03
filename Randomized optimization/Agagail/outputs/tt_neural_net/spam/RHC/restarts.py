import numpy as np
import matplotlib.pyplot as plt

filename1 = 'errors_6_1.csv'
f = open(filename1, 'r')
contents0 = f.read()
f.close()
splits = contents0.split('---')
for i, split in enumerate(splits):
    print (i, split)
    if i == 1:
        c0 = split.split('\n')

filename2 = 'errors_6_151.csv'
f = open(filename2, 'r')
contents1 = f.read()
f.close()
splits = contents1.split('---')
for i, split in enumerate(splits):
    print (i, split)
    if i == 1:
        c1 = split.split('\n')

filename3 = 'errors_6_301.csv'
f = open(filename3, 'r')
contents2 = f.read()
f.close()
splits = contents2.split('---')
for i, split in enumerate(splits):
    print (i, split)
    if i == 1:
        c2 = split.split('\n')

filename4 = 'errors_6_451.csv'
f = open(filename4, 'r')
contents3 = f.read()
f.close()
splits = contents3.split('---')
for i, split in enumerate(splits):
    print (i, split)
    if i == 1:
        c3 = split.split('\n')

n = []
r0 = []
r1 = []
r2 = []
r3 = []

for a, b, c,d in zip(c0,c1,c2,c3):
    r0_values = a.split(',')
    r1_values = b.split(',')
    r2_values = c.split(',')
    r3_values = d.split(',')
    if len(r0_values) < 4:
        continue
    if len(r1_values) < 4:
        continue
    if len(r2_values) < 4:
        continue
    if len(r3_values) < 4:
        continue
    n.append(r1_values[2])
    r0.append(r0_values[3])
    r1.append(r1_values[3])
    r2.append(r2_values[3])
    r3.append(r3_values[3])

fig = plt.figure()

plt.gca().set_color_cycle(['red', 'green', 'blue','black'])
plt.plot(n, r0)
plt.plot(n, r1)
plt.plot(n, r2)
plt.plot(n, r3)

plt.ylabel('Error')
plt.xlabel('Iterations')

algs = ['1', '151', '301', '451']
plt.legend(algs, loc='upper right')
plt.title('Error vs Iteration for RHC with Different Number of Restarts ')

fig.savefig('Plots/Error vs Iteration for RHC with Different Number of Restarts6.png')
plt.close()
