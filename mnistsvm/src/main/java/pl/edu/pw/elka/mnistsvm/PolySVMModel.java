package pl.edu.pw.elka.mnistsvm;

import org.nd4j.linalg.api.ndarray.INDArray;

public class PolySVMModel extends SvmModel {
   double n;

    public PolySVMModel(double n) {
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

    @Override
    void fillStats(ModelTestStats stats) {
        stats.svm="PolySvm";
        stats.sigma_r_n=n;
    }
}