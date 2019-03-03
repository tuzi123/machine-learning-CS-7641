package opt.ga;

import java.util.Arrays;
import java.util.List;

import util.linalg.Vector;
import opt.EvaluationFunction;
import shared.Instance;

/**
 * A Max K Color evaluation function
 * @author kmandal
 * @version 1.0
 */
public class MaxKColorFitnessFunction implements EvaluationFunction {
    
    public static int totalCounter = 0;
    /**
     * 
     */
    private Vertex[] vertices;
    private int graphSize;
    
    public MaxKColorFitnessFunction(Vertex[] vertices) {
        this.vertices = vertices;
        this.graphSize = vertices.length;
    }
    
    private boolean conflict = false;

    /**
     * @see opt.EvaluationFunction#value(opt.OptimizationData)
     * Find how many iterations does it take to find if k-colors can be or can not be assigned to a given graph.
     */
    public double value(Instance d) {
        totalCounter++;
        Vector data = d.getData();
        int n = data.size();
        double iterations = 0;
        conflict = false;
        // System.out.println("Sample color " + d.toString());
        for (int i = 0; i < n; i++) {
            int sampleColor = ((int) data.get(i));
            Vertex vertex = vertices[i];
            List<Integer> adjacencyColorMatrix = vertex.getAadjacencyColorMatrix();
            int l = adjacencyColorMatrix.size();
            for (int j = 0; j < l; j++) {
              int adjVertexIndex = adjacencyColorMatrix.get(j);
              int adjColor = (int) data.get(adjVertexIndex);
              if (sampleColor == adjColor) {
                conflict = true;
                break;
              }
              iterations++;
            }
        }
        return iterations;
    }

    
    public String foundConflict(){
    	return conflict ? "0" : "1";
    }
}
