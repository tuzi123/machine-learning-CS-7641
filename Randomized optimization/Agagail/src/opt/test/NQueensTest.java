package opt.test;

import java.util.Arrays;
import java.util.Random;
import opt.ga.NQueensFitnessFunction;
import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

import java.io.*;

/**
 * @author kmanda1
 * @version 1.0
 */
public class NQueensTest {
    
    /** The n value */
    static int N = 10;

    private static final String BASE_OUTPUT_DIR_PATH = "outputs/nqueens/";
    public static String getFullFileName(String fileName) {
        return BASE_OUTPUT_DIR_PATH + fileName;
    }
    
    public static void main(String[] args) {
        if (args.length > 0) {
            N = Integer.parseInt(args[0]);
        }

        String distanceResults = N + ",";
        String timeResults = N + ",";

        NQueensFitnessFunction ef = new NQueensFitnessFunction();
        Distribution odd = new DiscretePermutationDistribution(N);

        // Initializations for RHC and SA
        NeighborFunction nf = new SwapNeighbor();
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);

        // Randomized hill climbing
        long startTime = System.currentTimeMillis();
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 100);
        fit.train();
        timeResults += (System.currentTimeMillis() - startTime) + ",";
        distanceResults += ef.value(rhc.getOptimal()) + ",";

        System.out.println("RHC: " + ef.value(rhc.getOptimal()));
        // System.out.println("RHC: Board Position: ");
        // System.out.println(ef.boardPositions());

        
        System.out.println("Time : " + (System.currentTimeMillis() - startTime));
        System.out.println("============================");

        startTime = System.currentTimeMillis();
        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .1, hcp);
        fit = new FixedIterationTrainer(sa, 100);
        fit.train();
        timeResults += (System.currentTimeMillis() - startTime) + ",";
        distanceResults += ef.value(sa.getOptimal()) + ",";

        System.out.println("SA: " + ef.value(sa.getOptimal()));
        // System.out.println("SA: Board Position: ");
        // System.out.println(ef.boardPositions());
        
        System.out.println("Time : " + (System.currentTimeMillis() - startTime));
        System.out.println("============================");
        System.out.println(sa.getOptimal());
        
        // Genetic algorithm
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new SingleCrossOver();
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

        startTime = System.currentTimeMillis();
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 0, 10, gap);
        fit = new FixedIterationTrainer(ga, 100);
        fit.train();
        timeResults += (System.currentTimeMillis() - startTime) + ",";
        distanceResults += ef.value(ga.getOptimal()) + ",";

        System.out.println("GA: " + ef.value(ga.getOptimal()));
        // System.out.println("GA: Board Position: ");
        // System.out.println(ef.boardPositions());
        System.out.println("Time : " + (System.currentTimeMillis() - startTime));
        System.out.println("============================");
        
        // MIMIC
        Distribution df = new DiscreteDependencyTree(.1);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        startTime = System.currentTimeMillis();
        MIMIC mimic = new MIMIC(200, 10, pop);
        fit = new FixedIterationTrainer(mimic, 5);
        fit.train();
        timeResults += (System.currentTimeMillis() - startTime);
        distanceResults += ef.value(mimic.getOptimal());

        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
        // System.out.println("MIMIC: Board Position: ");
        // System.out.println(ef.boardPositions());
        System.out.println("Time : " + (System.currentTimeMillis() - startTime));


        try {
            String fileName = getFullFileName("optimal_value_analysis.csv");
            // PrintWriter writer = new PrintWriter(fileName, "UTF-8");
            PrintWriter writer = new PrintWriter(new FileOutputStream(new File(fileName), true)); 
            writer.println(distanceResults);
            writer.close();
        } catch(Exception e) {
            e.printStackTrace();
        }

        try {
            String fileName = getFullFileName("iteration_time_analysis.csv");
            // PrintWriter writer = new PrintWriter(fileName, "UTF-8");
            PrintWriter writer = new PrintWriter(new FileOutputStream(new File(fileName), true)); 
            writer.println(timeResults);
            writer.close();
        } catch(Exception e) {
            e.printStackTrace();
        }
    }
}
