package opt.test;

import java.util.Arrays;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.TwoSatEvaluationFunction;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

public class TwoSATTest {
	/** The number of variables */
    private static final int NUM_VARS = 8;
    /** The problem */
    /* Problem represented as a string (x1 V x2) ^ (x3 V not x4) as (x1 x2)|(x3 X4) */
	private static String problem = "";
	
	public static void main(String[] args) {
		problem =  "(x1 x4)|(x2 X4)|(x1 x3)|(x5 x7)|(x7 x0)|(x0 X2)|(x4 x6)|(X2 x5)|(x7 x1)|(x0 x3)|(x5 x4)|(x7 x3)|(x2 X4)|(x5 x6)|(x0 X1)|(x6 X7)|(X4 x2)|(X1 x5)|(x6 x1)|(x0 X7)|(X2 x7)|(x1 x5)|(x0 x6)|(x1 x0)|(x5 x7)|(x0 x2)|(x0 x3)|(x0 x4)|(x0 x5)|(x0 x6)|(x0 x7)|(x1 x2)|(x1 x3)|(x1 x4)|(x1 x6)|(x3 X2)|(X2 x4)|(X2 x5)|(X2 x6)|(X2 x7)|(x3 x4)|(x3 x5)|(x3 x6)|(x3 x7)|(X4 x5)|(X4 x6)|(X4 x7)|(x5 x7)|(x6 x7)|(X6 X7)|(X2 x1)|(X2 x0)|(X4 x0)|(X4 x1)|(X4 x2)|(X4 x3)|(X7 x0)|(X7 x1)|(X7 x3)|(X7 x2)|(X7 x3)|(X7 x4)|(X7 x5)";
		System.out.println("Ideal : " +  problem.split("\\|").length);
		int[] ranges = new int[NUM_VARS];
		Arrays.fill(ranges, 2);
        EvaluationFunction ef = new TwoSatEvaluationFunction(problem, NUM_VARS);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        fit.train();
        System.out.println(ef.value(rhc.getOptimal()));
        
        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
        fit = new FixedIterationTrainer(sa, 200000);
        fit.train();
        System.out.println(ef.value(sa.getOptimal()));
        
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 25, gap);
        fit = new FixedIterationTrainer(ga, 1000);
        fit.train();
        System.out.println(ef.value(ga.getOptimal()));
        
        MIMIC mimic = new MIMIC(200, 100, pop);
        fit = new FixedIterationTrainer(mimic, 1000);
        fit.train();
        System.out.println(ef.value(mimic.getOptimal()));
	}

}
