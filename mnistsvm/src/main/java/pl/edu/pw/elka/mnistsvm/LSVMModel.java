package pl.edu.pw.elka.mnistsvm;

import org.nd4j.linalg.api.ndarray.INDArray;

public class LSVMModel extends SvmModel {
    @Override
    double kernelFunction(INDArray inputVec, INDArray supprotVec) {
        int k=inputVec.columns();
        return inputVec.mmul(supprotVec.reshape(1,k).transpose()).getDouble(0,0);
    }
    @Override
    void fillStats(ModelTestStats stats) {
        stats.svm="LSvm";

    }
}
