package com.fitmymacros.imageclassifiermodel;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.util.Map;

public class ImageLoader {

    private static final int HEIGHT = 100;
    private static final int WIDTH = 100;
    private static final int CHANNELS = 3;

    public static INDArray loadImage(String imagePath) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
        INDArray image = loader.asMatrix(new File(imagePath));

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);

        return image;
    }

    public static void main(String[] args) throws IOException {
        Map<String, float[]> data = DataParser.parseMetaFile("train.txt");

        for (Map.Entry<String, float[]> entry : data.entrySet()) {
            String imagePath = entry.getKey();
            float[] nutritionalValues = entry.getValue();

            INDArray image = ImageLoader.loadImage(imagePath);
            // Use image and nutritionalValues in creating a DataSet, etc.
        }
    }

}
