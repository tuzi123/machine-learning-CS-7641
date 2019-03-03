package opt.test;

import java.util.Arrays;
import java.util.Random;

import opt.ga.MaxKColorFitnessFunction;
import opt.ga.Vertex;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

import java.io.*;

/**
 * 
 * @author kmandal
 * @version 1.0
 */
public class MaxKColoringTest {

    private static final String BASE_OUTPUT_DIR_PATH = "outputs/maxk/";
    public static String getFullFileName(String fileName) {
        return BASE_OUTPUT_DIR_PATH + fileName;
    }

    /** The n value */
    private static int N = 8; // number of vertices
    private static int L = 3; // L adjacent nodes per vertex
    private static int K = 5; // K possible colors
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        if (args.length > 0) 
            N = Integer.parseInt(args[0]);
        if (args.length > 1)
            L = Integer.parseInt(args[1]);
        if (args.length > 2)
            K = Integer.parseInt(args[2]);

        Random random = new Random();
        // create the random velocity
        Vertex[] vertices = new Vertex[N];
        for (int i = 0; i < N; i++) {
            Vertex vertex = new Vertex();
            vertices[i] = vertex;	
            vertex.setAdjMatrixSize(L);
            for (int j = 0; j < L; j++) {
                while (true) {
                    int randNum = random.nextInt(N);
                    if (randNum != i) {
                        if (vertex.getAadjacencyColorMatrix().contains(randNum)) {
                            continue;
                        }
                        vertex.getAadjacencyColorMatrix().add(randNum);
                        break;
                   }
                }
            }
        }
        for (int i = 0; i < N; i++) {
            Vertex vertex = vertices[i];
            System.out.println(Arrays.toString(vertex.getAadjacencyColorMatrix().toArray()));
        }
        // for rhc, sa, and ga we use a permutation based encoding

        String distanceResults = N + "," + L + "," + K + ",";
        String timeResults = N + "," + L + "," + K + ",";
        String correctResults = N + "," + L + "," + K + ",";

        int[] ranges = new int[N];
        Arrays.fill(ranges, K + 1);
        

        MaxKColorFitnessFunction ef = new MaxKColorFitnessFunction(vertices);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);


        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        long startTime = System.currentTimeMillis();
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 20000);
        fit.train();
        System.out.println("RHC: " + ef.value(rhc.getOptimal()));
        System.out.println(ef.foundConflict());
        System.out.println("Time : "+ (System.currentTimeMillis() - startTime));
        timeResults += (System.currentTimeMillis() - startTime) + ",";
        correctResults += ef.foundConflict()+ ",";

        System.out.println(rhc.getOptimal().toString());
        distanceResults += ef.value(rhc.getOptimal())+ ",";
        MaxKColorFitnessFunction.totalCounter = 0;
        System.out.println("============================");

        startTime = System.currentTimeMillis();
        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .8, hcp);
        fit = new FixedIterationTrainer(sa, 20000);
        fit.train();
        System.out.println("SA: " + ef.value(sa.getOptimal()));
        System.out.println(ef.foundConflict());
        System.out.println("Time : "+ (System.currentTimeMillis() - startTime));
        timeResults += (System.currentTimeMillis() - startTime) + ",";
        correctResults += ef.foundConflict()+ ",";
        
        System.out.println(sa.getOptimal().toString());
        System.out.println(MaxKColorFitnessFunction.totalCounter + "");
        distanceResults += ef.value(sa.getOptimal()) + ",";
        MaxKColorFitnessFunction.totalCounter = 0;
        System.out.println("============================");
        
        
        startTime = System.currentTimeMillis();
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 20, gap);
        fit = new FixedIterationTrainer(ga, 1000);
        fit.train();
        System.out.println("GA: " + ef.value(ga.getOptimal()));
        System.out.println(ef.foundConflict());
        System.out.println("Time : "+ (System.currentTimeMillis() - startTime));
        timeResults += (System.currentTimeMillis() - startTime) + ",";
        correctResults += ef.foundConflict()+ ",";
        
        System.out.println(ga.getOptimal().toString());
        System.out.println(MaxKColorFitnessFunction.totalCounter + "");
        distanceResults += ef.value(ga.getOptimal()) + ",";
        MaxKColorFitnessFunction.totalCounter = 0;
        System.out.println("============================");

        startTime = System.currentTimeMillis();
        MIMIC mimic = new MIMIC(200, 100, pop);
        fit = new FixedIterationTrainer(mimic, 100);
        fit.train();
        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));  
        System.out.println(ef.foundConflict());
        System.out.println("Time : "+ (System.currentTimeMillis() - startTime));
        timeResults += (System.currentTimeMillis() - startTime) + "";
        correctResults += ef.foundConflict()+ "";
        
        System.out.println(mimic.getOptimal().toString());
        System.out.println(MaxKColorFitnessFunction.totalCounter + "");
        distanceResults += ef.value(mimic.getOptimal()) + "";
        MaxKColorFitnessFunction.totalCounter = 0;
        System.out.println("============================");

        try {
             String fileName = getFullFileName("fitness_iteration_analysis_7.csv");
             // PrintWriter writer = new PrintWriter(fileName, "UTF-8");
             PrintWriter writer = new PrintWriter(new FileOutputStream(new File(fileName), true)); 
             writer.println(distanceResults);
             writer.close();
         } catch(Exception e) {
             e.printStackTrace();
         }

        try {
            String fileName = getFullFileName("iteration_time_analysis_7.csv");
            // PrintWriter writer = new PrintWriter(fileName, "UTF-8");
            PrintWriter writer = new PrintWriter(new FileOutputStream(new File(fileName), true)); 
            writer.println(timeResults);
            writer.close();
        } catch(Exception e) {
            e.printStackTrace();
        }
        
        try {
            String fileName = getFullFileName("number_correct_analysis_7.csv");
            PrintWriter writer = new PrintWriter(new FileOutputStream(new File(fileName), true)); 
            writer.println(correctResults);
            writer.close();
        } catch(Exception e) {
            e.printStackTrace();
        }
    }
}
