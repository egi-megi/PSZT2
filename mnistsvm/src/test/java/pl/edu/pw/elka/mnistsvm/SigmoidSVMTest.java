package pl.edu.pw.elka.mnistsvm;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.testng.annotations.Test;

import static org.testng.Assert.assertTrue;

public class SigmoidSVMTest {

    @Test
    void testTrivial() {
        for (int i=0 ; i<10; i++) {
            SvmModel.SEED=i;
            double[][] X = new double[][]{{1}, {-0.6}, {4}, {8}, {-4}, {-3}, {2}};
            double[] Y = new double[]{1, -1, 1, 1, -1, -1, 1};
            //double[][] X = new double[][]{{2, 1}, {4, 0}};
            //double[] Y = new double[]{-1,  1};

            SigmoidSVMModel mod = new SigmoidSVMModel();

            mod.svmTrain(Nd4j.createFromArray(X), Nd4j.createFromArray(Y), 1,0.001, 8);
            //Should be: 1, -1, 1, 1, -1, -1, 1, -1, 1
            double[][] testX = new double[][]{{1}, {-0.6}, {4}, {8}, {-4}, {-3}, {2}, {-5}, {5.5}};

            INDArray pre = mod.predict(Nd4j.createFromArray(testX));


            System.out.println(pre);
            System.out.println(mod.b);
            assertTrue(pre.getDouble(0) >= 0.0);
//            assertTrue(pre.getDouble(1) < 0.0);
        }
    }
}
