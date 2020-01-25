package pl.edu.pw.elka.mnistsvm;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.testng.annotations.Test;

import static org.testng.Assert.assertTrue;

public class PolySVMTest {

    @Test
    void testTrivial() {
        for (int i=0 ; i<10; i++) {
            SvmModel.SEED=i;
            double[][] X = new double[][]{{-0.1, 0.2}, {-0.6, 0.2}, {0.2, 0}, {0.3, 0.2}, {0.3, -0.4}, {0, 0.6}, {0.3, -0.5}};
            double[] Y = new double[]{1, -1, 1, 1, -1, -1, -1};
            //double[][] X = new double[][]{{2, 1}, {4, 0}};
            //double[] Y = new double[]{-1,  1};

            PolySVMModel mod = new PolySVMModel(4);

            mod.svmTrain(Nd4j.createFromArray(X), Nd4j.createFromArray(Y), 1,0.001, 8);
            //Should be: 1, -1, 1, 1, -1, -1, -1, 1, -1
            double[][] testX = new double[][]{{-0.1, 0.2}, {-0.6, 0.2}, {0.2, 0}, {0.3, 0.2}, {0.3, -0.4}, {0, 0.6}, {0.3, -0.5}, {0.1, -0.1}, {0.4, -0.4}};
            double[] expected= new double[]{1, -1, 1, 1, -1, -1, -1, 1, -1};
            INDArray pre = mod.predict(Nd4j.createFromArray(testX));

            //System.out.println();
            System.out.println(pre);
            System.out.println(mod.b);
            for (int j =0; j<expected.length;j++  ) {
                if (expected[j]>0) {
                    assertTrue(pre.getDouble(j) >= 0.0, "error on element "+j+" ");
                } else  {
                    assertTrue(pre.getDouble(j) < 0.0, "error on element "+j+" ");
                }
            }
//            assertTrue(pre.getDouble(0) >= 0.0);
//            assertTrue(pre.getDouble(1) < 0.0);
        }
    }
}
