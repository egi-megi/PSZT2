package pl.edu.pw.elka.mnistsvm;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.EqualsCondition;
import pl.edu.pw.elka.mnistsvm.reader.MnistMatrix;

import java.util.ArrayList;
import java.util.Random;

public abstract class SvmModel {
    static {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    INDArray supportVectorsX;
    INDArray supportVectorsY;
    INDArray alphas;
    double b;

    public static int SEED=-1;

    Random rnd = SEED==-1?new Random():new Random(SEED);



    abstract double kernelFunction(INDArray inputVec, INDArray supprotVec);


    /**
     * %SVMTRAIN Trains an SVM classifier using a simplified version of the SMO
     * %algorithm.
     * %   [model] = SVMTRAIN(X, Y, C, kernelFunction, tol, max_passes) trains an
     * %   SVM classifier and returns trained model. X is the matrix of training
     * %   examples.  Each row is a training example, and the jth column holds the
     * %   jth feature.  Y is a column matrix containing 1 for positive examples
     * %   and 0 for negative examples.  C is the standard SVM regularization
     * %   parameter.  tol is a tolerance value used for determining equality of
     * %   floating point numbers. max_passes controls the number of iterations
     * %   over the dataset (without changes to alpha) before the algorithm quits.
     * %
     * % Note: This is a simplified version of the SMO algorithm for training
     * %       SVMs. In practice, if you want to train an SVM classifier, we
     * %       recommend using an optimized package such as:
     * %
     * %           LIBSVM   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
     * %           SVMLight (http://svmlight.joachims.org/)
     * %
     * %
     *
     * @param X
     * @param Y
     * @param C
     * @param tol
     * @param max_passes
     */
    void svmTrain(INDArray X, INDArray Y, double C, double tol, int max_passes) {

// Data parameters
        int m = X.rows();

        int n = X.columns();
        System.out.println("Num rows: "+m+" num cols: "+n);

// Map 0 to -1
  //      Y.putWhere(0.0, -1, new EqualsCondition());


//% Variables
        alphas = Nd4j.zeros(m, 1);
        b = 0.0;
        INDArray E = Nd4j.zeros(m, 1);
        int passes = 0;
        double eta = 0;
        double L = 0;
        double H = 0;

//% Pre-compute the Kernel Matrix since our dataset is small
//% (in practice, optimized SVM packages that handle large datasets
//%  gracefully will _not_ do this)
//%
//% We have implemented optimized vectorized version of the Kernels here so
//                % that the svm training will run faster.
//    % Pre-compute the Kernel Matrix
//                % The following can be slow due to the lack of vectorization
        INDArray K = Nd4j.zeros(m,m);
        int[] index = new int[2];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                index[0] = i;
                index[1] = j;
                double val = kernelFunction(X.getRow(i),
                        X.getRow(j));
                K.putScalar(index, val);
                index[0] = j;
                index[1] = i;
                K.putScalar(index, val); //the matrix is symmetric

            }
        }

        //% Train
        System.out.println("\nTraining ...");
        int dots = 12;
        int j = 0;
        while (passes < max_passes) {

            int num_changed_alphas = 0;

            for (int i = 0; i < m; i++) {

                // % Calculate Ei = f(x(i)) - y(i) using (2).
                //         % E(i) = b + sum (X(i, :) * (repmat(alphas.*Y,1,n).*X)') - Y(i);
                E.putScalar(i, b + (alphas.transpose().mul(Y).mul(K.getColumn(i)).sum().getDouble(0)) - Y.getDouble(i));

                if ((Y.getDouble(i) * E.getDouble(i) < -tol && alphas.getDouble(i) < C) ||
                        (Y.getDouble(i) * E.getDouble(i) > tol && alphas.getDouble(i) > 0)) {

                    //In practice, there are many heuristics one can use to select the i and j. In this simplified code, we select them randomly.
                    j = rnd.nextInt(m);
                    while (j == i) { // % Make sure i \neq j
                        j = rnd.nextInt(m);
                    }

                    //       % Calculate Ej = f(x(j)) - y(j) using (2).
//                E(j) = b + sum (alphas.*Y.*K(:,j)) - Y(j);
                    E.putScalar(j, b + (alphas.transpose().mul(Y).mul(K.getColumn(j)).sum().getDouble(0)) - Y.getDouble(j));
                    //   % Save old alphas
                    double alpha_i_old = alphas.getDouble(i);
                    double alpha_j_old = alphas.getDouble(j);

//            % Compute L and H by (10) or (11).
                    if (Y.getDouble(i) == Y.getDouble(j)) {
                        L = Math.max(0, alphas.getDouble(j) + alphas.getDouble(i) - C);
                        H = Math.min(C, alphas.getDouble(j) + alphas.getDouble(i));
                    } else {
                        L = Math.max(0, alphas.getDouble(j) - alphas.getDouble(i));
                        H = Math.min(C, C + alphas.getDouble(j) - alphas.getDouble(i));
                    }

                    if (L != H) { //if l==h continue

                        //% Compute eta by (14).
                        eta = 2 * K.getDouble(i, j) - K.getDouble(i, i) - K.getDouble(j, j);
                        if (eta < 0) { //if (eta >= 0), %continue to next i.

                            //   %Compute and clip new value for alpha j using(12) and(15).

                            alphas.putScalar(j,
                                    alphas.getDouble(j) -
                                            (Y.getDouble(j) * (E.getDouble(i) - E.getDouble(j))) / eta);
                            //%Clip
                            alphas.putScalar(j, Math.min(H, alphas.getDouble(j)));
                            alphas.putScalar(j, Math.max(L, alphas.getDouble(j)));

                            //  %Check if change in alpha is significant
                            if (Math.abs(alphas.getDouble(j) - alpha_j_old) < tol) { //
//                %continue to next i.
//                %replace anyway
                                alphas.putScalar(j, alpha_j_old);

                            } else {

                                //   % Determine value for alpha i using(16).
                                alphas.putScalar(i, alphas.getDouble(i) +
                                        Y.getDouble(i) * Y.getDouble(j) * (alpha_j_old - alphas.getDouble(j)));

                                //  %Compute b1 and b2 using(17) and(18) respectively.
                                double b1 = b - E.getDouble(i)
                                        -Y.getDouble(i) * (alphas.getDouble(i) - alpha_i_old) * K.getDouble(i, j)
                                        - Y.getDouble(j) * (alphas.getDouble(j) - alpha_j_old) * K.getDouble(i, j);
                                double b2 = b - E.getDouble(j)
                                        -Y.getDouble(i) * (alphas.getDouble(i) - alpha_i_old) * K.getDouble(i, j)
                                        - Y.getDouble(j) * (alphas.getDouble(j) - alpha_j_old) * K.getDouble(j, j);

                                //     % Compute b by (19).
                                if (0 < alphas.getDouble(i) && alphas.getDouble(i) < C) {
                                    b = b1;
                                } else if (0 < alphas.getDouble(j) && alphas.getDouble(j) < C) {
                                    b = b2;
                                } else {
                                    b = (b1 + b2) / 2;
                                }

                                num_changed_alphas = num_changed_alphas + 1;
                            }
                        }
                    }
                }

            }

            if (num_changed_alphas == 0) {
                passes = passes + 1;
            } else {
                passes = 0;
            }

            System.out.print(".");


            dots = dots + 1;
            if (dots > 78) {
                dots = 0;
                System.out.println();
            }

        }
        System.out.println(" Done! \n\n");

//%Save the model
        ArrayList<Integer> al=new ArrayList<>();
        for (int i=0; i< m; i++) {
            if (alphas.getDouble(i)>0) {
                al.add(i);
            }
        }
//        idx = alphas > 0;

        int[]  idx=new int[al.size()];
        for (int i=0 ; i< idx.length; i++ ) {
            idx[i]=al.get(i);
        }
        supportVectorsX = X.getRows(idx);
        supportVectorsY = Y.reshape(m,1).getRows(idx).reshape(idx.length);
 //       model.kernelFunction = kernelFunction;
  // DOne       model.b = b;
        alphas = alphas.reshape(m,1).getRows(idx).reshape(idx.length);
   //Unsused      model.w = ((alphas. * Y) '*X)';

    }

    public void svmTrain(INDArray X, INDArray Y, double C) {
        svmTrain(X, Y, C, 100, 5);
    }


    public INDArray predict(INDArray X) {
        int m = X.rows();
        INDArray p = Nd4j.zeros(m);
        for (int i = 0; i < m; i++) {
            double prediction = 0;
            for (int j = 0; j < supportVectorsX.rows(); j++) {
                prediction = prediction +
                        alphas.getDouble(j)*(supportVectorsY.getDouble(j)) * (
                                kernelFunction(X.getRow(i), supportVectorsX.getRow(j)));
            }
            p.putScalar(i, prediction + b);
        }

        return p;
    }


}
