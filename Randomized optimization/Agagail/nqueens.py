import subprocess


for N in range(50, 350, 50):
    subprocess.call('java -cp ABAGAIL.jar opt.test.NQueensTest ' + str(N), shell=True)
