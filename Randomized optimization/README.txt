The assignment is based on Abagail using Java and Jython.
The Agagail libaray was modified to finish this assignment.
Folder Structure: (There are details in the outputs folder that are not shown)
---A2
   ---Agabail
      ---outputs
         ---knapsack 
            --- *.py (to generate plots)
            --- Plots
            --- *.csv
         ---maxk 
	 ---nqueens
         ---tsp
         ---tt_neural_net
            ---spam
            ---Pima
            ---wine
      ---src
      ---jython
      ---*.py (to run the problem)
   ---Results

In the Agagail folder, there are *.py files which are labeled by the name of the problem it's focused on.
---Traveling Salesman Problem, run tsp.py
---K coloring and Knapsack problems, run maxk_knapsack.py
---N queens probelm, run nqueens.py
---neural network probelm, run neural_net*.py for different datasets.
   ---in the neural_net_spam.py, it tested all the algorithms and there parameters influence for spam data, 
      some part is commented not to be run, in order to get the result for a specific part, 
      uncomment the commmand, each command is clearly labeled for what problem it is working on.

In the outputs folder under Agagail, it contains the result. After running the code, all result will be saved in .csv files.
Open the folder where each specific data files are located, there are .py files which can generate the plots. 
The Plots will be saved in a plots folder which is located at the same place with the .py files.
For some of the .py files that generate plots, the results need to be copied from the data file to the .py files to initiate the values.

