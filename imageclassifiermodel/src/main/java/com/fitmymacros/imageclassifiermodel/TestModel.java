package com.fitmymacros.imageclassifiermodel;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;

public class TestModel {

    public static void main(String[] args) throws IOException {
        // Load the trained model from the zip file
        File modelFile = new File("trainedModel.zip");
        MultiLayerNetwork model = MultiLayerNetwork.load(modelFile, true);

        // Path to the new image you want to test
        String imagePath = "images/apple_pie/1014775.jpg"; // Replace with your test image path

        // Load and preprocess the new image
        INDArray image = ImageLoader.loadImage(imagePath);

        // Make predictions
        INDArray output = model.output(image);

        // Extract the predicted nutritional values
        float predictedCalories = output.getFloat(0);
        float predictedFat = output.getFloat(1);
        float predictedProtein = output.getFloat(2);
        float predictedCarbs = output.getFloat(3);

        // Print the predicted nutritional values
        System.out.println("Predicted Calories: " + predictedCalories);
        System.out.println("Predicted Fat: " + predictedFat);
        System.out.println("Predicted Protein: " + predictedProtein);
        System.out.println("Predicted Carbs: " + predictedCarbs);
    }
}
