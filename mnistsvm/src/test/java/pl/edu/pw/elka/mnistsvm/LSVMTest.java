package pl.edu.pw.elka.mnistsvm;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.testng.annotations.Test;

import static org.testng.Assert.*;

public class LSVMTest {

    @Test
    void testKernel() {
        LSVMModel mod=new LSVMModel();
    INDArray fst= Nd4j.ones(1,3);
    INDArray sec= Nd4j.ones(1,3);
    assertEquals(mod.kernelFunction(fst,sec),3.0);
    sec.putScalar(2,10.0);
    System.out.println(sec);
    }


    @Test
    void testTrivial() {

        double[][] X= new double[][]{{1,0},{2,1},{4,0},{5,1}};
        double[] Y = new double[]{-1,-1,1,1};

        LSVMModel mod=new LSVMModel();

        mod.svmTrain(Nd4j.createFromArray(X),Nd4j.createFromArray(Y),0.1);

        double [][]testX=new double[][]{{6,1},{0.0,0.0}};

        INDArray pre=mod.predict(Nd4j.createFromArray(testX));


        assertTrue(pre.getDouble(0)>=0.0);
        assertTrue(pre.getDouble(0)>=0.0);
    }


}
