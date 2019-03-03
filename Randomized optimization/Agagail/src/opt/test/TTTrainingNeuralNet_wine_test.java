package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;
import util.linalg.Vector;

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
public class TTTrainingNeuralNet_wine_test {

    private static final String BASE_OUTPUT_DIR_PATH = "outputs/tt_neural_net/wine/";
    public static String getFullFileName(String fileName) {
        return BASE_OUTPUT_DIR_PATH + fileName;
    }

    private static final int VERSION_NO = 4;
    private static Instance[] instances_train = initializeInstancesTrain();
    private static Instance[] instances_test = initializeInstancesTest();
    private static int INPUT_LAYERS = 11, HIDDEN_LAYERS = 2, OUTPUT_LAYER = 9, TRAINING_ITERATIONS = 200;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();    
    private static ErrorMeasure measure = new SumOfSquaresError();
    private static DataSet set = new DataSet(instances_train);
    
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
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 10, 100, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
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
                predicted = instances_test[j].getLabel().getData().argMax();
                actual = networks[i].getOutputValues().argMax();
                double trash = predicted == actual ? correct++ : incorrect++;
            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10, 9);

            displayResults += oaNames[i] + "," + TRAINING_ITERATIONS + "," +  VERSION_NO +
                                "," + correct + "," + incorrect + "," +
                                df.format(correct/(correct+incorrect)*100) + "," +
                                df.format(trainingTime) + "," + df.format(testingTime) + "\n";

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        }
        try {
            String fileName = getFullFileName("outputs_" + VERSION_NO+".csv");
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

            double error = 0,predicted, actual, correct=0, incorrect=0;
            for(int j = 0; j < instances_train.length; j++) {
                network.setInputValues(instances_train[j].getData());
                network.run();

                Instance output = instances_train[j].getLabel(), example = new Instance(network.getOutputValues());
                //example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                example.setLabel(new Instance(network.getOutputValues()));
                error += measure.value(output, example);
                
                predicted = instances_train[j].getLabel().getData().argMax();
                actual = network.getOutputValues().argMax();
                double trash = predicted == actual ? correct++ : incorrect++;
                

                
            }
            errorResults += oaName + "," + TRAINING_ITERATIONS + "," + i + "," + df.format(error) + ","+df.format(correct/(correct+incorrect)*100)+"\n";
            // System.out.println(df.format(error));
        }
        try {
            String fileName = getFullFileName("errors_" +  VERSION_NO + "_" + TRAINING_ITERATIONS + ".csv");
            PrintWriter writer = new PrintWriter(new FileOutputStream(new File(fileName), true));
            writer.println("---");
            writer.print(errorResults);
            writer.close();
        } catch(Exception e) {
            e.printStackTrace();
        }
    }

    private static Instance[] initializeInstancesTrain() {
        double[][][] attributes = new double[999][][];
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
            //System.out.println("value = " + c);
            double[] classes = new double[9];
            int a = c-1;
            classes[a] = 1.0;
            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
            //instances[i].setLabel(new Instance(attributes[i][1][0] < 15 ? 0 : 1));
            instances[i].setLabel(new Instance(classes));
        }       
        return instances;
    }
    
    
    private static Instance[] initializeInstancesTest() {
        double[][][] attributes = new double[600][][];
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
            //System.out.println("value = " + c);
            double[] classes = new double[9];
            int a = c-1;
            classes[a] = 1.0;
            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
            //instances[i].setLabel(new Instance(attributes[i][1][0] < 15 ? 0 : 1));
            instances[i].setLabel(new Instance(classes));
        }       
        return instances;
    }
    
    
}
