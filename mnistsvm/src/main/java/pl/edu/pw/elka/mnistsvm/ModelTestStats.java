package pl.edu.pw.elka.mnistsvm;

public class ModelTestStats {
    int testSize=1000;
    int trainingSize;
    String metaModel;
    String svm;
    double sigma_r_n=-1;
    double gamma=-1;
    double tol;
    double C;
    int maxPasses;
    long trainingTime;
    long testTime;
    double accuracy;
    double precision;
    double recall;
    double f1;

    static String getHeader(){
        return "metaModel;svm;sigma_r_n;gamma;tol;C;maxPasses;trainingTime;testTime;accuracy;precision;recall;f1";
    }
    String csvString (){
        return  metaModel + ";" + svm + ";" + sigma_r_n + ";" + gamma + ";" + tol + ";" + C + ";" + maxPasses +
                ";" + trainingTime + ";" + testTime + ";" + accuracy + ";" + precision + ";" + recall + ";" + f1;
    }

}
