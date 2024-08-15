package com.fitmymacros.imageclassifiermodel;

import java.io.*;
import java.nio.file.*;
import java.util.*;

public class DataParser {
    private static final String IMAGE_DIR = "/Users/pablodelalamo/Documents/FitMyMacros/ImageClassificationModel/images/";
    private static final String META_DIR = "/Users/pablodelalamo/Documents/FitMyMacros/ImageClassificationModel/meta/";

    public static Map<String, float[]> parseMetaFile(String fileName) throws IOException {
        Map<String, float[]> dataMap = new HashMap<>();
        List<String> lines = Files.readAllLines(Paths.get(META_DIR + fileName));
        for (String line : lines) {
            String[] parts = line.split(",");
            String imagePath = IMAGE_DIR + parts[0] + ".jpg";
            float[] nutritionalInfo = new float[] {
                    Float.parseFloat(parts[1]), // Calories
                    Float.parseFloat(parts[2]), // Fat
                    Float.parseFloat(parts[3]), // Protein
                    Float.parseFloat(parts[4]) // Carbs
            };
            dataMap.put(imagePath, nutritionalInfo);
        }
        return dataMap;
    }

    public static void main(String[] args) throws IOException {
        Map<String, float[]> trainData = parseMetaFile("train.txt");
        Map<String, float[]> testData = parseMetaFile("test.txt");

        for (Map.Entry<String, float[]> entry : trainData.entrySet()) {
            String key = entry.getKey();
            float[] value = entry.getValue();
            System.out.println(key + " -> " + Arrays.toString(value));
        }

    }
}
