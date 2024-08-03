package com.fitmymacros.imageclassifiermodel;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class EvaluateModel {

    public void evaluate(MultiLayerNetwork model, DataSetIterator testIterator) {
        System.out.println("Evaluating model...");
        Evaluation eval = model.evaluate(testIterator);
        System.out.println(eval.stats());
    }
}
