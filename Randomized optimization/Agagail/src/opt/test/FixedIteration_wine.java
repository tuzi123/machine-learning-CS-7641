package opt.test;

import opt.OptimizationAlgorithm;
import opt.RHC;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.Scanner;
import func.nn.NeuralNetwork;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;
import util.linalg.Vector;


public class FixedIteration_wine {
	private static final String BASE_OUTPUT_DIR_PATH = "outputs/tt_neural_net/wine/fixed_iteration/";
    public static String getFullFileName(String fileName) {
        return BASE_OUTPUT_DIR_PATH + fileName;
    }

    private static final int VERSION_NO = 1;
 
    private static Instance[] instances_train = initializeInstancesTrain();
    private static Instance[] instances_test = initializeInstancesTest();

    private static int INPUT_LAYERS = 11, HIDDEN_LAYERS = 2, OUTPUT_LAYER = 9, TRAINING_ITERATIONS = 1;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();
    private static DataSet set_train = new DataSet(instances_train);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "", displayResults = "";

    private static DecimalFormat df = new DecimalFormat("0.000");
    
    
    public static void main(String[] args) {
        if (args.length > 0) 
        	TRAINING_ITERATIONS = Integer.parseInt(args[0]);

        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {INPUT_LAYERS, HIDDEN_LAYERS, OUTPUT_LAYER});
            nnop[i] = new NeuralNetworkOptimizationProblem(set_train, networks[i], measure);
        }
        
        oa[0] = new RHC(nnop[0],1);
        oa[1] = new SimulatedAnnealing(1E11, .5, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 10, 100, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
        	double start = System.nanoTime(), end, trainingTime, testingTime;
            train(oa[i], networks[i], oaNames[i]); // trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10, 9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual,test_error,correct_test=0, incorrect_test=0;
            start = System.nanoTime();
            for(int j = 0; j < instances_test.length; j++) {
                networks[i].setInputValues(instances_test[j].getData());
                networks[i].run();
                predicted = instances_test[j].getLabel().getData().argMax();
                actual = networks[i].getOutputValues().argMax();
                double trash = predicted == actual ? correct_test++ : incorrect_test++;
            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10, 9); 
            test_error = correct_test/(correct_test+incorrect_test);
          
            double correct_train=0, incorrect_train=0, train_error;
            for(int j = 0; j < instances_train.length; j++) {
                networks[i].setInputValues(instances_train[j].getData());
                networks[i].run();
                predicted = instances_train[j].getLabel().getData().argMax();
                actual = networks[i].getOutputValues().argMax();
                double trash = predicted == actual ? correct_train++ : incorrect_train++;
                
            }
            train_error = correct_train/(correct_train+incorrect_train);
            

            displayResults += oaNames[i] + "," + TRAINING_ITERATIONS + "," +  VERSION_NO +
                                ","  + correct_test + "," + incorrect_test + "," +
                                df.format(test_error*100) + "," +
                                correct_train + "," + incorrect_train + "," +
                                df.format(train_error*100) + "," +
                                df.format(trainingTime) + "," + df.format(testingTime) + "\n";

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified test " + correct_test + " instances." +
                    "\nIncorrectly classified test " + incorrect_test + " instances.\nPercent correctly classified test: "
                    + df.format(test_error*100) 
                    + ": \nCorrectly classified train " + correct_train + " instances." +
                    "\nIncorrectly classified train " + incorrect_train + " instances.\nPercent correctly classified train: "
                    + df.format(train_error*100) 
                    + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        }
        try {
            String fileName = getFullFileName("outputs_" +  VERSION_NO + ".csv");
            PrintWriter writer = new PrintWriter(new FileOutputStream(new File(fileName), true));
            writer.print(displayResults);
            writer.close();
        } catch(Exception e) {
            e.printStackTrace();
        }
        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        for(int i = 0; i < TRAINING_ITERATIONS; i++) {
        	oa.train();
        }     
    }

    
    private static Instance[] initializeInstancesTrain() {
        final int NO_INSTANCES = 999;
        double[][][] attributes = new double[NO_INSTANCES][][];
        final int ATT_LENGTH = INPUT_LAYERS;
        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("winequality-red-train.csv")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(";");
                attributes[i] = new double[2][];
                attributes[i][0] = new double[11]; // 11 attributes
                attributes[i][1] = new double[1];
                for(int j = 0; j < 11; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());
                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];
        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            int c = (int) attributes[i][1][0];
            double[] classes = new double[9];
            int a = c-1;
            classes[a] = 1.0;
            instances[i].setLabel(new Instance(classes));
        }
        return instances;
    }
    
    private static Instance[] initializeInstancesTest() {
        final int NO_INSTANCES = 600;
        double[][][] attributes = new double[NO_INSTANCES][][];
        final int ATT_LENGTH = INPUT_LAYERS;
        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("winequality-red-test.csv")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(";");
                attributes[i] = new double[2][];
                attributes[i][0] = new double[11]; // 11 attributes
                attributes[i][1] = new double[1];
                for(int j = 0; j < 11; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());
                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];
        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            int c = (int) attributes[i][1][0];
            double[] classes = new double[9];
            int a = c-1;
            classes[a] = 1.0;
            instances[i].setLabel(new Instance(classes));
        }
        return instances;
    }
    
    private static boolean isEqualOutputs(Vector actual, Vector predicted) {
        int max_at = 0;
        double max = 0;
       // Where the actual max should be
        int actual_index = 0;
        for (int i = 0; i < actual.size(); i++) 
        {
          double aVal = actual.get(i);

          if (aVal == 1.0) {
            actual_index = i;}
          double bVal = predicted.get(i);

          if (bVal > max) {
            max = bVal;
            max_at = i;}
        }
        return actual_index == max_at;
      }
    

}
