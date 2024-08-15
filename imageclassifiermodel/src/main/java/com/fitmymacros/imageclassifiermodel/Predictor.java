package com.fitmymacros.imageclassifiermodel;

import java.io.IOException;
import java.util.Map;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Predictor {

    public static float[] predictNutrition(MultiLayerNetwork model, String imagePath) throws IOException {
        INDArray image = ImageLoader.loadImage(imagePath);
        INDArray output = model.output(image);
        return output.toFloatVector();
    }

    public static void main(String[] args) throws IOException {
        MultiLayerNetwork model = CNNModel.createModel();
        // Load and train the model before this step

        Map<String, float[]> testData = DataParser.parseMetaFile("test.txt");

        for (Map.Entry<String, float[]> entry : testData.entrySet()) {
            String imagePath = entry.getKey();
            float[] expectedNutrition = entry.getValue();

            float[] predictedNutrition = predictNutrition(model, imagePath);

            System.out.println("Image: " + imagePath);
            System.out.println("Expected Calories: " + expectedNutrition[0] + ", Predicted: " + predictedNutrition[0]);
            System.out.println("Expected Fat: " + expectedNutrition[1] + ", Predicted: " + predictedNutrition[1]);
            System.out.println("Expected Protein: " + expectedNutrition[2] + ", Predicted: " + predictedNutrition[2]);
            System.out.println("Expected Carbs: " + expectedNutrition[3] + ", Predicted: " + predictedNutrition[3]);
        }
    }
}
