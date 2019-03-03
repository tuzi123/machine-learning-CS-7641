package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class GA_Mutation_test {

    private static final String BASE_OUTPUT_DIR_PATH = "outputs/tt_neural_net/spam/GA_test/";
    public static String getFullFileName(String fileName) {
        return BASE_OUTPUT_DIR_PATH + fileName;
    }

    private static final int VERSION_NO = 1;
 
    private static Instance[] instances_train = initializeInstancesTrain();
    private static Instance[] instances_test = initializeInstancesTest();

    private static int INPUT_LAYERS = 57, HIDDEN_LAYERS = 2, OUTPUT_LAYER = 1, TRAINING_ITERATIONS = 400;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();
    private static int mutation_rate= 10;
    private static int population = 200;
    private static int mutation;
    
    private static DataSet set_train = new DataSet(instances_train);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[1];
    private static String[] oaNames = {"GA"};
    private static String results = "", displayResults = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        if (args.length > 0) 
            mutation_rate = Integer.parseInt(args[0]);

        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {INPUT_LAYERS, HIDDEN_LAYERS, OUTPUT_LAYER});
            nnop[i] = new NeuralNetworkOptimizationProblem(set_train, networks[i], measure);
        }
        mutation = (int) mutation_rate * population/100;
        oa[0] = new StandardGeneticAlgorithm(population, mutation, 100, nnop[0]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct_test = 0, incorrect_test = 0;
            train(oa[i], networks[i], oaNames[i]); // trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10, 9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < instances_test.length; j++) {
                networks[i].setInputValues(instances_test[j].getData());
                networks[i].run();
                actual = Double.parseDouble(instances_test[j].getLabel().toString());
                predicted = Double.parseDouble(networks[i].getOutputValues().toString());
                actual = (actual >= 0.5 ? 1: 0);
                predicted = (predicted >= 0.5 ? 1: 0);
                //System.out.println("predicted test " + predicted + " actual test "+ actual+"\n---------------------------");
                double trash = Math.abs(predicted - actual) < 0.2 ? correct_test++ : incorrect_test++;
            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10, 9); 
            
            double correct_train=0, incorrect_train=0;
            for(int j = 0; j < instances_train.length; j++) {
                networks[i].setInputValues(instances_train[j].getData());
                networks[i].run();
                actual = Double.parseDouble(instances_train[j].getLabel().toString());
                predicted = Double.parseDouble(networks[i].getOutputValues().toString());
                actual = (actual >= 0.5 ? 1: 0);
                predicted = (predicted >= 0.5 ? 1: 0);
                //System.out.println("predicted test " + predicted + " actual test "+ actual+"\n---------------------------");
                double trash = Math.abs(predicted - actual) < 0.2 ? correct_train++ : incorrect_train++;
            }

            displayResults += oaNames[i] + "," + mutation + "," +  VERSION_NO +
                                ","  + correct_test + "," + incorrect_test + "," +
                                df.format(correct_test/(correct_test+incorrect_test)*100) + "," +
                                correct_train + "," + incorrect_train + "," +
                                df.format(correct_train/(correct_train+incorrect_train)*100) + "," +
                                df.format(trainingTime) + "," + df.format(testingTime) + "\n";

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified test " + correct_test + " instances." +
                    "\nIncorrectly classified test " + incorrect_test + " instances.\nPercent correctly classified test: "
                    + df.format(correct_test/(correct_test+incorrect_test)*100) 
                    + ": \nCorrectly classified train " + correct_train + " instances." +
                    "\nIncorrectly classified train " + incorrect_train + " instances.\nPercent correctly classified train: "
                    + df.format(correct_train/(correct_train+incorrect_train)*100) 
                    + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        }
        try {
            String fileName = getFullFileName("outputs_GA_" +  VERSION_NO + ".csv");
            PrintWriter writer = new PrintWriter(new FileOutputStream(new File(fileName), true));
            writer.print(displayResults);
            writer.close();
        } catch(Exception e) {
            e.printStackTrace();
        }
        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");
        String errorResults = "";
        for(int i = 0; i < TRAINING_ITERATIONS; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances_train.length; j++) {
                network.setInputValues(instances_train[j].getData());
                network.run();
                Instance output = instances_train[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);               
            }
            //error =correct/(correct+incorrect)*100;
            errorResults += oaName + "," + TRAINING_ITERATIONS + "," + i + "," + df.format(error) + "\n";
            // System.out.println(df.format(error));
        }
        try {
            String fileName = getFullFileName("errors_train_test_" +  VERSION_NO + "_" + mutation + ".csv");
            PrintWriter writer = new PrintWriter(new FileOutputStream(new File(fileName), true));
            writer.println("---");
            writer.print(errorResults);
            writer.close();
        } catch(Exception e) {
            e.printStackTrace();
        }
    }

    
    private static Instance[] initializeInstancesTrain() {
        final int NO_INSTANCES = 2830;
        double[][][] attributes = new double[NO_INSTANCES][][];
        final int ATT_LENGTH = INPUT_LAYERS;
        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("spambase_train.data")));
            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");
                attributes[i] = new double[2][];
                attributes[i][0] = new double[ATT_LENGTH]; // 11 attributes
                attributes[i][1] = new double[1]; // 1 output - class

                // writing input values
                for(int j = 0; j < ATT_LENGTH; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());
                // writing output value
                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        Instance[] instances = new Instance[attributes.length];
        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0] >= 1 ? 1: 0));
        }
        return instances;
    }
    
    private static Instance[] initializeInstancesTest() {
        final int NO_INSTANCES = 1771;
        double[][][] attributes = new double[NO_INSTANCES][][];
        final int ATT_LENGTH = INPUT_LAYERS;
        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("spambase_test.data")));
            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");
                attributes[i] = new double[2][];
                attributes[i][0] = new double[ATT_LENGTH]; // 11 attributes
                attributes[i][1] = new double[1]; // 1 output - class

                // writing input values
                for(int j = 0; j < ATT_LENGTH; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());
                // writing output value
                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        Instance[] instances = new Instance[attributes.length];
        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0] >= 1 ? 1: 0));
        }
        return instances;
    }
    
}
