package com.fitmymacros.imageclassifiermodel;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.learning.config.Adam;

public class CNNModel {

    public static MultiLayerNetwork createModel() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.001))
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(3)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(3, new OutputLayer.Builder(LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nOut(4)
                        .build())
                .setInputType(InputType.convolutional(100, 100, 3))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));
        return model;
    }

    public static void main(String[] args) throws IOException {
        System.out.println("Entering main");
        MultiLayerNetwork model = createModel();
        System.out.println("created the model");
        // Load the data
        Map<String, float[]> trainData = DataParser.parseMetaFile("train.txt");
        System.out.println("parsed train data");

        // Convert it to DataSetIterator (Assuming you have written a utility for this)
        DataSetIterator trainIterator = DataSetUtility.createIterator(trainData);
        System.out.println("created the iterator");

        // Train the model
        for (int i = 0; i < 10; i++) {
            System.out.println("Training model. Epoch " + i);
            model.fit(trainIterator);
        }

        // Optionally, save the model after training
        model.save(new File("trainedModel.zip"));
    }
}
