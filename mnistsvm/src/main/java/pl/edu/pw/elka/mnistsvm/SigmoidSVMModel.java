package pl.edu.pw.elka.mnistsvm;

import org.nd4j.linalg.api.ndarray.INDArray;

public class SigmoidSVMModel extends SvmModel {
    double r;
    double gamma;

    public SigmoidSVMModel(double r, double gamma) {
        this.r = r;
        this.gamma = gamma;
    }

    public SigmoidSVMModel(){
        this(2.0, 1.0);
    }
    @Override
    double kernelFunction(INDArray inputVec, INDArray supprotVec) {
        int k=inputVec.columns();
        return Math.tanh((inputVec.mmul(supprotVec.reshape(1,k).transpose()).getDouble(0,0)) * gamma + r);
    }

    @Override
    void fillStats(ModelTestStats stats) {
        stats.svm="SigmoidSvm";
        stats.sigma_r_n=r;
        stats.gamma=gamma;
    }
}
