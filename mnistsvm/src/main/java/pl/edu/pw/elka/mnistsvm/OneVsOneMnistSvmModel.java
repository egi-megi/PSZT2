package pl.edu.pw.elka.mnistsvm;

import pl.edu.pw.elka.mnistsvm.reader.MnistMatrix;

import java.util.Arrays;
import java.util.stream.IntStream;

public class OneVsOneMnistSvmModel extends MnistModel {

    int C=1;
    int trainSize=180;
    SvmModelCreator creator;



    public OneVsOneMnistSvmModel(SvmModelCreator creator) {
        this.creator = creator;
    }

    @Override
    public String getName() {
        return "OneVsOneMnist "+creator.modelName();
    }


    SvmModel[][] models=new SvmModel[10][10];

    @Override
    protected void doTrainingWithSetUpTrainLables() {

        IntStream.of(0,1,2,3,4,5,6,7,8).parallel().forEach(i-> {
            for (int j = i + 1; j < 10; j++) {

                ProbeForTraining pft = new ProbeForTraining();
                addWithLabelsToProbe(i, trainSize / 2, pft);
                addWithLabelsToProbe(j, trainSize / 2, pft);
                models[i][j] = creator.createModel();
                TrainData data = pft.getTrainData(i);
                models[i][j].svmTrain(data.X, data.y, C);
            }

        });
    }




    @Override
    public void test(MnistMatrix[] matrix) {
        long testStart=System.currentTimeMillis();
        Arrays.stream(matrix).parallel().forEach(m -> {
                 int size=m.getData().columns()*m.getData().rows();
                int[] votes = new int[10];

                for (int i=0 ; i<9; i++) {
                    for (int j=i+1; j<10; j++) {
                        double p=models[i][j].predict(m.getData().reshape(1,size)).getDouble(0);
                        if (p>0.0) {
                          //simple voting
                            votes[i]++;
                        } else {
                            votes[j]++;
                        }
                    }
                }

                int max=0;
                int maxInd=0;
                for (int i=0;i<10; i++) {
                    if (votes[i]>max) {
                        max=votes[i];
                        maxInd=i;
                    }
                }
                m.setInferencedLabel(maxInd);
            });
        long endTime=System.currentTimeMillis();

        double d=((double)matrix.length) / (((double)endTime-testStart)/1000.0);
        System.out.println("Inference time is: "+((endTime-testStart)/1000)+"."+((endTime-testStart)%1000)+"s");
        System.out.println("Efficincy : "+(d)+" tests/s");
    }
}
