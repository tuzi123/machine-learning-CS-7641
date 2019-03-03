import subprocess


for N in range(1, 101, 5):
    subprocess.call('java -cp ABAGAIL.jar opt.test.TravelingSalesmanTest ' + str(N), shell=True)
