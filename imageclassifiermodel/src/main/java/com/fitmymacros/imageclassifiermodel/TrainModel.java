package com.fitmymacros.imageclassifiermodel;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class TrainModel {

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("Please provide the path to the dataset as an argument.");
            System.exit(1);
        }

        String dataPath = args[0];
        TrainModel trainModel = new TrainModel();
        trainModel.train(dataPath);
    }

    public void train(String dataPath) throws Exception {
        int height = 224;
        int width = 224;
        int channels = 3;
        int numClasses = 101; // Number of meal categories
        int numAttributes = 4; // Calories, protein, carbs, fat
        int batchSize = 16;
        int epochs = 10;

        DataPreparation dataPreparation = new DataPreparation();
        DataSetIterator trainIterator = dataPreparation.prepareData(dataPath + "/images", dataPath + "/meta/train.txt",
                batchSize, height, width, channels, numClasses);
        DataSetIterator testIterator = dataPreparation.prepareData(dataPath + "/images", dataPath + "/meta/test.txt",
                batchSize, height, width, channels, numClasses);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.001))
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(numAttributes)
                        .activation(Activation.IDENTITY)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        for (int i = 0; i < epochs; i++) {
            System.out.println("Epoch " + (i + 1) + "/" + epochs);
            model.fit(trainIterator);
        }

        model.save(new File("food_classifier_model.zip"), true);

        // Evaluate the model
        evaluateModel(model, testIterator);
    }

    private void evaluateModel(MultiLayerNetwork model, DataSetIterator testIterator) {
        System.out.println("Evaluating model...");
        Evaluation eval = model.evaluate(testIterator);
        System.out.println(eval.stats());
    }
}
