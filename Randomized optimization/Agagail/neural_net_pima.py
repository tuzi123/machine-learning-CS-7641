import subprocess

# iteration and time analysis of all three algorithms: spam
for N in range(50, 750, 100):
    subprocess.call('java -cp ABAGAIL.jar opt.test.NN_Pima ' + str(N), shell=True)
# -Dpython.console.encoding=UTF-8


# cooling rate analysis for SA: spam
#for cool_rate in range(9, 100, 5):    
#    subprocess.call('java -cp ABAGAIL.jar opt.test.SA ' + str(cool_rate), shell=True)
    
# iteration and time analysis of all three algorithms: wime
#for N in range(50, 750, 100):
#    subprocess.call('java -cp ABAGAIL.jar opt.test.TTTrainingNeuralNet ' + str(N), shell=True)
    
# mutation analysis of GA: spam
#for N in range(10, 50, 10):
#    subprocess.call('java -cp ABAGAIL.jar opt.test.GA_Mutation ' + str(N), shell=True)
    
# mutation analysis of GA: spam
#for N in range(1, 15, 3):
#    subprocess.call('java -cp ABAGAIL.jar opt.test.RHC_Restarts ' + str(N), shell=True)
    
# mutation analysis of GA: spam
#for N in range(100, 450, 100):
#    subprocess.call('java -cp ABAGAIL.jar opt.test.GA_Population ' + str(N), shell=True)
    
    
    