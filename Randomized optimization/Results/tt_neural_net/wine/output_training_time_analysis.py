time_values = '''RHC,50,3,635.0,964.0,39.712,0.259,0.005
SA,50,3,390.0,1209.0,24.390,0.275,0.004
GA,50,3,786.0,813.0,49.156,9.794,0.003
RHC,150,3,637.0,962.0,39.837,0.752,0.005
SA,150,3,329.0,1270.0,20.575,0.774,0.003
GA,150,3,680.0,919.0,42.527,28.669,0.003
RHC,250,3,638.0,961.0,39.900,1.205,0.005
SA,250,3,53.0,1546.0,3.315,1.257,0.003
GA,250,3,681.0,918.0,42.589,47.700,0.003
RHC,350,3,683.0,916.0,42.714,1.685,0.005
SA,350,3,0.0,1599.0,0.000,1.691,0.003
GA,350,3,680.0,919.0,42.527,66.383,0.003
RHC,450,3,681.0,918.0,42.589,2.125,0.004
SA,450,3,168.0,1431.0,10.507,2.301,0.004
GA,450,3,708.0,891.0,44.278,85.382,0.004
RHC,550,3,823.0,776.0,51.470,2.530,0.005
SA,550,3,10.0,1589.0,0.625,2.590,0.003
GA,550,3,699.0,900.0,43.715,105.427,0.003
RHC,650,3,776.0,823.0,48.530,3.052,0.005
SA,650,3,642.0,957.0,40.150,3.059,0.003
GA,650,3,681.0,918.0,42.589,123.079,0.003'''


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
