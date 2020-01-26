package pl.edu.pw.elka.mnistsvm;

import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import pl.edu.pw.elka.mnistsvm.reader.MnistDataReader;
import pl.edu.pw.elka.mnistsvm.reader.MnistMatrix;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;


public class ModelTrainAndTest {
    static {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    static ArrayList<Model> models = new ArrayList<>();

    static {
         /*
           Baseline returning the most probable class
           new BaselineMaxClassModel(),
          */
         /*
         one vs all on LSVM
         new OneVsAllMnistSvmModel(new OneVsAllMnistSvmModel.SvmModelCreator() {
                @Override
                public SvmModel createModel() {
                    return new LSVMModel();
                }

                @Override
                public String modelName() {
                    return "LSVMModel";
                }
            }),
            */
        for (int C : new int[]{10}) {
            for (double tol : new double[]{0.0000001}) {
                for (int maxPasses : new int[]{5}) {
                    for (int trainingSize : new int[]{300}) {

                        models.add(new OneVsAllMnistSvmModel(new SvmModelCreator() {
                            @Override
                            public SvmModel createModel() {
                                return new LSVMModel();
                            }

                            @Override
                            public String modelName() {
                                return "LSVMModel";
                            }
                        }, C, trainingSize, tol, maxPasses));

                        for (double sigma : new double[]{0.5}) {
                            models.add(new OneVsAllMnistSvmModel(new SvmModelCreator() {
                                @Override
                                public SvmModel createModel() {
                                    return new RBFSVMModel(sigma);
                                }

                                @Override
                                public String modelName() {
                                    return "RBFSModel";
                                }
                            }, C, trainingSize, tol, maxPasses));

                        }

                        for (double n : new double[]{2}) {
                            models.add(new OneVsAllMnistSvmModel(new SvmModelCreator() {
                                @Override
                                public SvmModel createModel() {
                                    return new PolySVMModel(n);
                                }

                                @Override
                                public String modelName() {
                                    return "PolySVMModel";
                                }
                            }, C, trainingSize, tol, maxPasses));

                        }

                        for (double r: new double[]{1.0}) {
                            for (double gamma: new double[]{0.5}){
                                models.add(new OneVsAllMnistSvmModel(new SvmModelCreator() {
                                    @Override
                                    public SvmModel createModel() {
                                        return new SigmoidSVMModel(r, gamma);
                                    }

                                    @Override
                                    public String modelName() {
                                        return "SigmoidSVMModel";
                                    }
                                }, C, trainingSize, tol, maxPasses));
                            }
                        }


                    }


                    for (int trainingSize : new int[]{100}) {

                        models.add(new OneVsOneMnistSvmModel(new SvmModelCreator() {
                            @Override
                            public SvmModel createModel() {
                                return new LSVMModel();
                            }

                            @Override
                            public String modelName() {
                                return "LSVMModel";
                            }
                        }, C, trainingSize, tol, maxPasses));

                        for (double sigma : new double[]{0.5}) {
                            models.add(new OneVsOneMnistSvmModel(new SvmModelCreator() {
                                @Override
                                public SvmModel createModel() {
                                    return new RBFSVMModel(sigma);
                                }

                                @Override
                                public String modelName() {
                                    return "RBFSModel";
                                }
                            }, C, trainingSize, tol, maxPasses));
                        }

                        for (double n : new double[]{2}) {
                            models.add(new OneVsOneMnistSvmModel(new SvmModelCreator() {
                                @Override
                                public SvmModel createModel() {
                                    return new PolySVMModel(n);
                                }

                                @Override
                                public String modelName() {
                                    return "PolySVMModel";
                                }
                            }, C, trainingSize, tol, maxPasses));

                        }

                        for (double r: new double[]{1.0}) {
                            for (double gamma: new double[]{0.5}){
                                models.add(new OneVsOneMnistSvmModel(new SvmModelCreator() {
                                    @Override
                                    public SvmModel createModel() {
                                        return new SigmoidSVMModel(r, gamma);
                                    }

                                    @Override
                                    public String modelName() {
                                        return "SigmoidSVMModel";
                                    }
                                }, C, trainingSize, tol, maxPasses));
                            }
                        }

                    }
                }
            }
        }
    }

    static MnistMatrix[] mnistMatrix ;//= new MnistDataReader().readData("data" + File.separator+"train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
    static MnistMatrix[] testMnistMatrix ;
    public static ModelTestStats testSingleModel(Model m) {
        ModelTestStats stats = new ModelTestStats();
        try {
           m.train(mnistMatrix, stats);


            int testSize = 500;

            MnistMatrix[] smallTest = new MnistMatrix[testSize];
            stats.testSize = testSize;
            for (int i = 0; i < smallTest.length; i++) {
                smallTest[i] = testMnistMatrix[i];
            }
            m.test(smallTest, stats);
            System.out.println("Model: " + m.getName());
            printPrecRecall(smallTest, stats);
        } catch (Exception e) {
            e.printStackTrace();
            stats.precision = Double.NaN;
        }
        return stats;
    }


    public static void printPrecRecall(MnistMatrix[] matrices, ModelTestStats stats) {

        int[][] realInt = new int[matrices.length][10];
        int[][] predInt = new int[matrices.length][10];

        for (int i = 0; i < matrices.length; i++) {
            realInt[i][matrices[i].getLabel()] = 1;
            predInt[i][matrices[i].getInferencedLabel()] = 1;
        }

        Evaluation ev = new Evaluation(10);
        ev.eval(Nd4j.createFromArray(realInt), Nd4j.createFromArray(predInt));
        stats.precision = ev.precision();
        stats.recall = ev.recall();
        stats.accuracy = ev.accuracy();
        stats.f1 = ev.f1();
        System.out.println(ev.stats());

    }


    public static void main(String[] args) throws IOException {
        FileWriter fw = new FileWriter("test-eval-" + System.currentTimeMillis() + ".csv");
        fw.write(ModelTestStats.getHeader());
        fw.write("\n");
        fw.flush();
        mnistMatrix = new MnistDataReader().readData("data" + File.separator+"train-images.idx3-ubyte", "data"+File.separator+"train-labels.idx1-ubyte");
        testMnistMatrix = new MnistDataReader().readData("data" + File.separator + "t10k-images.idx3-ubyte", "data"+File.separator+"t10k-labels.idx1-ubyte");

        for (Model m : models) {
            ModelTestStats stat = testSingleModel(m);
            fw.write(stat.csvString());
            fw.write("\n");
            fw.flush();
        }
        fw.close();
    }


}
