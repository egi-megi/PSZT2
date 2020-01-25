package pl.edu.pw.elka.mnistsvm;

import org.nd4j.linalg.api.ndarray.INDArray;

public class PolySVMModel extends SvmModel {
   int n;

    public PolySVMModel(int n) {
        this.n = n;
    }

    public PolySVMModel(){
        this(2);
    }

    @Override
    double kernelFunction(INDArray inputVec, INDArray supprotVec) {
        int k=inputVec.columns();
        return Math.pow((1 + inputVec.mmul(supprotVec.reshape(1,k).transpose()).getDouble(0,0)), n);
    }

}