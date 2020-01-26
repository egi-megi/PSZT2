package pl.edu.pw.elka.mnistsvm;


import pl.edu.pw.elka.mnistsvm.reader.MnistMatrix;

public interface Model {


    String getName();
    void train(MnistMatrix[] matrix,ModelTestStats testStats);
    void test(MnistMatrix[] matrix,ModelTestStats testStats);

}
