package com.fitmymacros.imageclassifiermodel;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

public class DataSetUtility {

    private static final int batchSize = 8;

    public static DataSetIterator createIterator(Map<String, float[]> dataMap) throws IOException {
        List<DataSet> dataSets = new ArrayList<>();

        for (Map.Entry<String, float[]> entry : dataMap.entrySet()) {
            String imagePath = entry.getKey();
            float[] nutritionalValues = entry.getValue();

            INDArray image = ImageLoader.loadImage(imagePath);
            INDArray labels = Nd4j.create(nutritionalValues).reshape(1, 4);
            ;

            DataSet dataSet = new DataSet(image, labels);
            dataSets.add(dataSet);
        }

        return new ListDataSetIterator<>(dataSets, batchSize);
    }
}
