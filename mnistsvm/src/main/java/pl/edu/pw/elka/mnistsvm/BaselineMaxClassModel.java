package pl.edu.pw.elka.mnistsvm;

import pl.edu.pw.elka.mnistsvm.reader.MnistMatrix;


public class BaselineMaxClassModel implements Model {



    int mostProbbaleLabel;

    public String getName() {
        return "MostOftenClassModel";
    }

    public void train(MnistMatrix[] matrix,ModelTestStats testStats) {
        int[] labelsCount=new int[10];
        for (MnistMatrix m:matrix) {
            labelsCount[m.getLabel()]++;
        }
        int maxV=0;
        int maxI=0;
        for (int i=0 ; i<10;i++) {
            if (labelsCount[i]>maxV) {
                maxV=labelsCount[i];
                maxI=i;
            }
        }
        mostProbbaleLabel=maxI;
    }

    public void test(MnistMatrix[] matrix,ModelTestStats testStats) {
        for (MnistMatrix m:matrix) {
            m.setInferencedLabel(mostProbbaleLabel);

        }
    }
}


