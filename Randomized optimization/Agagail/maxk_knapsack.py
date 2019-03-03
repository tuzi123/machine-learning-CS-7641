import subprocess

# run maxk
for N in range(5, 50, 1):
    subprocess.call('java -cp ABAGAIL.jar opt.test.MaxKColoringTest ' + str(N), shell=True)

# run knapsack
for N in range(5, 50, 1):
    subprocess.call('java -cp ABAGAIL.jar opt.test.KnapsackTest ' + str(N), shell=True)
