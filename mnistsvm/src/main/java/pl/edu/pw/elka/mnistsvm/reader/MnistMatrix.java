package pl.edu.pw.elka.mnistsvm.reader;

import org.nd4j.linalg.api.ndarray.INDArray;


public class MnistMatrix {

    private INDArray data;

    private int nRows;
    private int nCols;

    private int label;
    private int inferencedLabel;

    public MnistMatrix(INDArray data, int label) {
        this.nRows = data.rows();
        this.nCols = data.columns();
        this.data=data;

        this.label=label;
    }


    public INDArray getData() {
        return data;
    }



    public int getLabel() {
        return label;
    }

    public int getInferencedLabel() {
        return inferencedLabel;
    }

    public void setInferencedLabel(int inferencedLabel) {
        this.inferencedLabel = inferencedLabel;
    }

    public int getNumberOfRows() {
        return nRows;
    }

    public int getNumberOfColumns() {
        return nCols;
    }

}
