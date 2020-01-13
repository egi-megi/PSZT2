package pl.edu.pw.elka.mnistsvm;


import pl.edu.pw.elka.mnistsvm.reader.MnistMatrix;

public interface Model {


    String getName();
    void train(MnistMatrix[] matrix);
    void test(MnistMatrix[] matrix);

}
