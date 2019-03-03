package opt.example;

import opt.EvaluationFunction;
import shared.Instance;
import util.linalg.Vector;

public class TwoSatEvaluationFunction implements EvaluationFunction {
	/* Problem represented as a string (x0 V x1) ^ (x3 V not x4) as (x0 x1)|(x3 X4) */
	String problem = "";
	/* Assignment represented as 100..10 where 1 is true and 0 is false*/
	Vector data;
	/* The number of variables*/
	int vars = 0;
	
	public TwoSatEvaluationFunction(String problem, int number) {
        this.problem = problem;
        System.out.println(problem);
        vars = number;
    }
	
	@Override
	public double value(Instance d) {
		data = d.getData();
		//d.length should be equal to number of variables
		String[] clauses = problem.split("\\|");
		double value = 0;
		for (String clause : clauses) {
			String component1 = clause.split(" ")[0].substring(1,3);
			String component2 = clause.split(" ")[1].substring(0,2);
			boolean component1Check1 = component1.startsWith("x") && (data.get((int)(component1.charAt(1))-48) == 1.0);
			boolean component1Check2 = component1.startsWith("X") && (data.get((int)(component1.charAt(1))-48) == 0.0);
			boolean component2Check1 = component2.startsWith("x") && (data.get((int)(component2.charAt(1))-48) == 1.0);
			boolean component2Check2 = component2.startsWith("X") && (data.get((int)(component2.charAt(1))-48) == 0.0);
			boolean c1 = component1Check1 || component1Check2; 
			boolean c2 = component2Check1 || component2Check2;
			if (c1 || c2) { 
				value += 1;
			}
		}
		double penalty = (clauses.length - value);
		return value-penalty;
	}

}
