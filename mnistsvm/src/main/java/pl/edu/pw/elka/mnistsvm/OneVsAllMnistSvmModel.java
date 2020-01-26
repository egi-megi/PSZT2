package pl.edu.pw.elka.mnistsvm;

import pl.edu.pw.elka.mnistsvm.reader.MnistMatrix;

import java.util.Arrays;
import java.util.stream.IntStream;

public class OneVsAllMnistSvmModel extends MnistModel {


    SvmModelCreator creator;


    public OneVsAllMnistSvmModel(SvmModelCreator creator,int c, int trainSize, double tol, int maxPasses) {
        super(c,trainSize,tol,maxPasses);
        this.creator = creator;
    }

    @Override
    public String getName() {
        return "OneVsAllMnist "+creator.modelName();
    }


    SvmModel[] models=new SvmModel[10];

    @Override
    protected void doTrainingWithSetUpTrainLables(ModelTestStats testStats) {
        long testStart=System.currentTimeMillis();

        testStats.metaModel="OneVsAll";

        IntStream.of(0,1,2,3,4,5,6,7,8,9).parallel().forEach(i->{
                ProbeForTraining pft=new ProbeForTraining();
                addWithLabelsToProbe(i,trainSize/2,pft);
                for (int j=0;j<10; j++) {
                    if (j!=i) {
                        addWithLabelsToProbe(j,trainSize/18,pft);
                    }
                }
                models[i]=creator.createModel();
                TrainData data=pft.getTrainData(i);
                models[i].svmTrain(data.X, data.y, C,tol,maxPasses);
            });
        models[0].fillStats(testStats);
        long endTime=System.currentTimeMillis();
        testStats.trainingTime=endTime-testStart;
        System.out.println("Training time is: "+((endTime-testStart)/1000)+"."+((endTime-testStart)%1000)+"s");
    }

    @Override
    public void test(MnistMatrix[] matrix,ModelTestStats testStats) {
        long testStart=System.currentTimeMillis();
        Arrays.stream(matrix).parallel().forEach(m -> {
               double max=-Double.MAX_VALUE;
                int maxInd=-1;
                int size=m.getData().columns()*m.getData().rows();
                for (int i=0;i<10; i++) {
                    double p=models[i].predict(m.getData().reshape(1,size)).getDouble(0);
                    if (p>max) {
                        max=p;
                        maxInd=i;
                    }
                }
                m.setInferencedLabel(maxInd);
            });
        long endTime=System.currentTimeMillis();

        double d=((double)matrix.length) / (((double)endTime-testStart)/1000.0);
        testStats.testTime=endTime-testStart;
        System.out.println("Inference time is: "+((endTime-testStart)/1000)+"."+((endTime-testStart)%1000)+"s");
        System.out.println("Efficincy : "+(d)+" tests/s");
    }
}
