import subprocess

# iteration and time analysis of all three algorithms: spam
#for N in range(50, 750, 100):
#    subprocess.call('java -cp ABAGAIL.jar opt.test.NueralNetSpam ' + str(N), shell=True)
# -Dpython.console.encoding=UTF-8

# cooling rate analysis for SA: spam
#for cool_rate in range(9, 100, 5):    
#    subprocess.call('java -cp ABAGAIL.jar opt.test.SA ' + str(cool_rate), shell=True)
    
# iteration and time analysis of all three algorithms: wine
#for N in range(50, 750, 100):
#    subprocess.call('java -cp ABAGAIL.jar opt.test.TTTrainingNeuralNet ' + str(N), shell=True)
    
# mutation analysis of GA: spam
#for N in range(10, 50, 10):
#    subprocess.call('java -cp ABAGAIL.jar opt.test.GA_Mutation ' + str(N), shell=True)
    
# restarts analysis of RHC: spam
for N in range(1, 600, 150):
    subprocess.call('java -cp ABAGAIL.jar opt.test.RHC_Restarts ' + str(N), shell=True)
    
# population analysis of GA: spam
#for N in range(1, 16, 3):
#    subprocess.call('java -cp ABAGAIL.jar opt.test.GA_Population ' + str(N), shell=True)

# mutation probability analysis of GA: spam
#for N in range(10, 50, 10):
#    subprocess.call('java -cp ABAGAIL.jar opt.test.GA_Mutation_test ' + str(N), shell=True)

# crossover analysis of GA: spam
#for N in range(10, 100, 20):
#    subprocess.call('java -cp ABAGAIL.jar opt.test.GA_crossover ' + str(N), shell=True)
    
    
# neural net: fixed iteration analysis:spam
#for N in range(1, 70, 1):
 #   subprocess.call('java -cp ABAGAIL.jar opt.test.NN_FixedIteration ' + str(N), shell=True)
    
# neural net: fixed iteration analysis:wine
#for N in range(1, 60, 1):
#    subprocess.call('java -cp ABAGAIL.jar opt.test.FixedIteration_wine ' + str(N), shell=True)