package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import shared.Instance;

import java.io.*;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    /** The n value */
    static int N = 50;

    private static final String BASE_OUTPUT_DIR_PATH = "outputs/tsp/";
    public static String getFullFileName(String fileName) {
        return BASE_OUTPUT_DIR_PATH + fileName;
    }
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        if (args.length > 0) {
            N = Integer.parseInt(args[0]);
        }
        Instance optimal;
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        
        int maxX = 300;
        int maxY = 300;

        for (int i = 0; i < points.length; i++) {
            int randX = random.nextInt(maxX);
            int randY = random.nextInt(maxY);

            points[i][0] = (double) randX;
            points[i][1] = (double) randY;   
            System.out.println(points[i][0] + "," + points[i][1]);
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        
        long startTime = System.currentTimeMillis();
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        fit.train();

        String rhcTime = (System.currentTimeMillis() - startTime) + "";

        optimal = rhc.getOptimal();
        System.out.println(optimal.toString());
        double rhcDistance = 1/ef.value(optimal);
        // System.out.println(1/ef.value(optimal));

        startTime = System.currentTimeMillis();
        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
        fit = new FixedIterationTrainer(sa, 200000);
        fit.train();

        String saTime = (System.currentTimeMillis() - startTime) + "";

        optimal = sa.getOptimal();
        System.out.println(optimal.toString());
        double saDistance = 1/ef.value(optimal);
        // System.out.println(1/ef.value(optimal));
        
        startTime = System.currentTimeMillis();
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 40, gap);
        fit = new FixedIterationTrainer(ga, 1000);
        fit.train();

        String gaTime = (System.currentTimeMillis() - startTime) + "";

        optimal = ga.getOptimal();
        System.out.println(optimal.toString());
        double gaDistance = 1/ef.value(optimal);
        // System.out.println(1/ef.value(optimal));
        
        // for mimic we use a sort encoding
        ef = new TravelingSalesmanRouteEvaluationFunction(points);
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        odd = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        startTime = System.currentTimeMillis();
        MIMIC mimic = new MIMIC(200, 100, pop);
        fit = new FixedIterationTrainer(mimic, 1000);
        fit.train();

        String mimicTime = (System.currentTimeMillis() - startTime) + "";

        optimal = mimic.getOptimal();
        System.out.println(optimal.toString());
        double mimicDistance = 1/ef.value(optimal);
        // System.out.println(mimicDistance);

        try {
            String distanceResults = N + "," + rhcDistance + "," + saDistance + "," + gaDistance + "," + mimicDistance;
            String fileName = getFullFileName("iteration_distance_analysis.csv");
            // PrintWriter writer = new PrintWriter(fileName, "UTF-8");
            PrintWriter writer = new PrintWriter(new FileOutputStream(new File(fileName), true)); 
            writer.println(distanceResults);
            writer.close();
        } catch(Exception e) {
            e.printStackTrace();
        }

        try {
            String timeResults = N + "," + rhcTime + "," + saTime + "," + gaTime + "," + mimicTime;
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
