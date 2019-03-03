import subprocess

for N in range(50, 500, 50):
    subprocess.call('java -cp ABAGAIL.jar opt.test.TTTrainingNeuralNet_wine_test ' + str(N), shell=True)
# -Dpython.console.encoding=UTF-8