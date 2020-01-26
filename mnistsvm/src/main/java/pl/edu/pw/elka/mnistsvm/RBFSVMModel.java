package pl.edu.pw.elka.mnistsvm;

import org.nd4j.linalg.api.ndarray.INDArray;

public class RBFSVMModel extends SvmModel {
    double sigma;

    public RBFSVMModel(double sigma) {
        this.sigma = sigma;
    }

    public RBFSVMModel(){
        this(0.5);
    }
    @Override
    double kernelFunction(INDArray inputVec, INDArray supprotVec) {
        int k=inputVec.columns();
        INDArray substracted = inputVec.sub(supprotVec);
        double vectorLength = substracted.mmul(substracted.reshape(1,k).transpose()).getDouble(0);
        //return 1;
        return Math.exp(-vectorLength/(2* sigma * sigma));
        //return inputVec.mmul(supprotVec.reshape(1,k).transpose()).getDouble(0,0);
    }

    @Override
    void fillStats(ModelTestStats stats) {
        stats.svm="RBF SVM";
        stats.sigma_r_n=sigma;
    }

}