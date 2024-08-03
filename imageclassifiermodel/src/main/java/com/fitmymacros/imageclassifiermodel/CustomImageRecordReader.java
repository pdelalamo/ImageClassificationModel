package com.fitmymacros.imageclassifiermodel;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.api.writable.DoubleWritable;
import org.nd4j.common.primitives.Pair;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

public class CustomImageRecordReader extends ImageRecordReader {

    private Map<String, List<Writable>> imageAttributesMap;
    private Iterator<Pair<Writable, List<Writable>>> iterator;

    public CustomImageRecordReader(int height, int width, int channels, ParentPathLabelGenerator labelMaker,
            String attributesFilePath) throws IOException {
        super(height, width, channels, labelMaker);
        imageAttributesMap = new HashMap<>();
        loadAttributes(attributesFilePath);

        List<Pair<Writable, List<Writable>>> pairs = new ArrayList<>();
        for (Map.Entry<String, List<Writable>> entry : imageAttributesMap.entrySet()) {
            // Create a Pair with Text cast to Writable
            Pair<Writable, List<Writable>> pair = new Pair<>((Writable) new Text(entry.getKey()), entry.getValue());
            pairs.add(pair);
        }
        iterator = pairs.iterator();
    }

    private void loadAttributes(String attributesFilePath) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(attributesFilePath));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] parts = line.split(",");
            if (parts.length != 6)
                continue; // Assumes format: imageFileName, calories, protein, carbs, fat

            String imageFileName = parts[0];
            List<Writable> attributes = new ArrayList<>();
            attributes.add(new Text(parts[1])); // label
            attributes.add(new DoubleWritable(Double.parseDouble(parts[2]))); // calories
            attributes.add(new DoubleWritable(Double.parseDouble(parts[3]))); // protein
            attributes.add(new DoubleWritable(Double.parseDouble(parts[4]))); // carbs
            attributes.add(new DoubleWritable(Double.parseDouble(parts[5]))); // fat

            imageAttributesMap.put(imageFileName, attributes);
        }
        reader.close();
    }

    @Override
    public List<Writable> next() {
        if (iterator.hasNext()) {
            Pair<Writable, List<Writable>> currentPair = iterator.next();
            String imagePath = currentPair.getFirst().toString();
            List<Writable> attributes = currentPair.getSecond();

            try {
                // Set current file to the image path
                setCurrentFile(new File(imagePath));

                // Load the image data
                List<Writable> imageData = super.next();
                imageData.addAll(attributes); // Append attributes to the image data
                return imageData;
            } catch (Exception e) {
                System.err.println("Error reading image: " + imagePath);
                e.printStackTrace();
                if (hasNext()) {
                    return next(); // Try to read the next image
                } else {
                    throw new RuntimeException("No more images to read.", e);
                }
            }
        }
        throw new NoSuchElementException("No more data to read.");
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }
}