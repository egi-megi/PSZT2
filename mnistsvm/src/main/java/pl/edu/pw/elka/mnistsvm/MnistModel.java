package pl.edu.pw.elka.mnistsvm;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import pl.edu.pw.elka.mnistsvm.reader.MnistMatrix;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.stream.Collectors;

public abstract class MnistModel implements  Model {

    static {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }


    ArrayList<INDArray>[] trainForLabels=new ArrayList[10];


    protected static class TrainData {

        INDArray X;
        INDArray y;

        public TrainData(INDArray x, INDArray y) {
            X = x;
            this.y = y;
        }
    }

    protected static class ProbeForTraining{

         ArrayList<INDArray>[] selected=new ArrayList[10];
        {
            for (int i=0 ; i<10; i++) {
                selected[i]=new ArrayList<>();
            }
        }
         public void add(int label, Collection<INDArray> elems){
             selected[label].addAll(elems);
         }


         public TrainData getTrainData(int posInt) {
             ArrayList<Pair<INDArray, Double>> data=new ArrayList<>();

             for (int i=0; i<10;  i++) {
                 double val=i==posInt?1.0:-1.0;
                 data.addAll(selected[i].parallelStream().map(x-> Pair.of(x,val)).collect(Collectors.toList()));
             }
             Collections.shuffle(data);
             int size=data.get(0).getFirst().columns()*data.get(0).getFirst().rows();
             INDArray X=Nd4j.create(data.size(),size);
             double[] y=new double[data.size()];
             for (int i=0; i<data.size(); i++) {
                 X.putRow(i,data.get(i).getFirst().reshape(1,size));
                 y[i]=data.get(i).getSecond();
             }
             return new TrainData(X, Nd4j.createFromArray(y));


         }

    }


    void addWithLabelsToProbe(int label, int size, ProbeForTraining probe){
       synchronized (trainForLabels[label]) {
           Collections.shuffle(trainForLabels[label]);
           probe.add(label, trainForLabels[label].subList(0, size));
       }
    }




    protected abstract void doTrainingWithSetUpTrainLables();


    @Override
    public void train(MnistMatrix[] matrix) {
        trainForLabels=new ArrayList[10];
        for (int i=0; i<10 ; i++) {
            trainForLabels[i]=new ArrayList<>();
        }
        for (MnistMatrix m:matrix) {
            trainForLabels[m.getLabel()].add(m.getData());
        }
        doTrainingWithSetUpTrainLables();
    }

}
