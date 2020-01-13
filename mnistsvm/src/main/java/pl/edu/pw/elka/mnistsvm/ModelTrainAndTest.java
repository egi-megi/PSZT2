package pl.edu.pw.elka.mnistsvm;

import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import pl.edu.pw.elka.mnistsvm.reader.MnistDataReader;
import pl.edu.pw.elka.mnistsvm.reader.MnistMatrix;

import java.io.IOException;


public class ModelTrainAndTest {

    static Model[] models={
            new BaselineMaxClassModel()
    };


    public static void main(String[] args) throws IOException {
        for (Model m:models) {
            MnistMatrix[] mnistMatrix = new MnistDataReader().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
            m.train(mnistMatrix);
            mnistMatrix = new MnistDataReader().readData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
            m.test(mnistMatrix);
            System.out.println("Model: "+m.getName());
            printPrecRecall(mnistMatrix);
        }
    }


    public static void printPrecRecall(MnistMatrix[] matrices) {

        int[][] realInt=new int[matrices.length][10];
        int[][] predInt=new int[matrices.length][10];

        for (int i=0; i< matrices.length; i++) {
            realInt[i][matrices[i].getLabel()]=1;
            predInt[i][matrices[i].getInferencedLabel()]=1;
        }

        Evaluation ev=new Evaluation(10);
        ev.eval(Nd4j.createFromArray(realInt), Nd4j.createFromArray(predInt));
        System.out.println(ev.stats());

    }


}
