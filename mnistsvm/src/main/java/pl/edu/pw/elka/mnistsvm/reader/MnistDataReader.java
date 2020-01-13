package pl.edu.pw.elka.mnistsvm.reader;

import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

public class MnistDataReader  {

    public MnistMatrix[] readData(String dataFilePath, String labelFilePath) throws IOException {

        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataFilePath)));
        int magicNumber = dataInputStream.readInt();
        int numberOfItems = dataInputStream.readInt();
        int nRows = dataInputStream.readInt();
        int nCols = dataInputStream.readInt();

        System.out.println("magic number is " + magicNumber);
        System.out.println("number of items is " + numberOfItems);
        System.out.println("number of rows is: " + nRows);
        System.out.println("number of cols is: " + nCols);

        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilePath)));
        int labelMagicNumber = labelInputStream.readInt();
        int numberOfLabels = labelInputStream.readInt();

        System.out.println("labels magic number is: " + labelMagicNumber);
        System.out.println("number of labels is: " + numberOfLabels);

        MnistMatrix[] data = new MnistMatrix[numberOfItems];

        assert numberOfItems == numberOfLabels;
        int[][] array=new int[nRows][nCols];
        for(int i = 0; i < numberOfItems; i++) {
            int label=labelInputStream.readUnsignedByte();
            for (int r = 0; r < nRows; r++) {
                for (int c = 0; c < nCols; c++) {
                    array[r][c]= dataInputStream.readUnsignedByte();
                }
            }
            data[i] = new MnistMatrix(Nd4j.createFromArray(array),label);

        }
        dataInputStream.close();
        labelInputStream.close();
        return data;
    }
}
