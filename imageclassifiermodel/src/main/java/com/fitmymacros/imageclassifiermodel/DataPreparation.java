package com.fitmymacros.imageclassifiermodel;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.util.Random;

public class DataPreparation {

    public DataSetIterator prepareData(String imagesPath, String labelsPath, int batchSize, int height, int width,
            int channels, int numClasses) throws Exception {
        // Use FileSplit to handle image files
        FileSplit fileSplit = new FileSplit(new File(imagesPath), NativeImageLoader.ALLOWED_FORMATS, new Random(123));
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        // Custom record reader to handle additional attributes
        CustomImageRecordReader recordReader = new CustomImageRecordReader(height, width, channels, labelMaker,
                labelsPath);
        recordReader.initialize(fileSplit);

        RecordReaderDataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1,
                numClasses);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        dataSetIterator.setPreProcessor(scaler);

        return dataSetIterator;
    }
}
