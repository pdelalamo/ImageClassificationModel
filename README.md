Food Image Nutritional Information Estimator

Project Overview
The Food Image Nutritional Information Estimator is a deep learning model designed to predict the nutritional content of food items based on images. The model outputs four key nutritional values: Calories, Fat, Protein, and Carbohydrates. It is built using a Convolutional Neural Network (CNN) architecture implemented with the Deeplearning4j library. This project includes data preprocessing, model training, evaluation, and prediction.


Project Structure

com.fitmymacros.imageclassifiermodel/
│
├── CNNModel.java                # Defines and trains the CNN model
├── DataParser.java              # Parses metadata and image paths for training/testing
├── DataSetUtility.java          # Converts data to DataSetIterator for training
├── EvaluateModel.java           # Evaluates the model's performance on test data
├── ImageLoader.java             # Loads and preprocesses images
├── Predictor.java               # Makes predictions based on the trained model
├── TestModel.java               # Tests the model with new images


Requirements

Java 8+
Deeplearning4j 1.0.0-beta7
Nd4j 1.0.0-beta7
Datavec 1.0.0-beta7
Install the necessary dependencies using your preferred package management tool (e.g., Maven or Gradle).


Installation

Clone the repository:
git clone https://github.com/pdelalamo/ImageClassificationModel.git
cd ImageClassificationModel
Configure the Project:

Update the paths in DataParser.java and TestModel.java to point to the correct directories for images and metadata on your local system.


Build the Project:
Build the project using your preferred build tool (e.g., Maven).

Dataset Preparation
Images: Place your images in the directory specified in IMAGE_DIR in DataParser.java. Ensure that the images are named correctly as per the metadata file.

Metadata: Prepare a train.txt and test.txt file in the format:
image_name,calories,fat,protein,carbs
For example:
apple_pie,250,10,3,34
The paths to these files should be updated in DataParser.java.

Model Training
The CNN model is defined and trained in CNNModel.java. To start training:

Ensure that the train.txt file is properly prepared.
Run the CNNModel.main() method to start the training process. The training process involves the following steps:
Parse training data.
Convert data to DataSetIterator.
Train the model for the specified number of epochs (10 in my case as 101 categories with 2000 images per category can be considered a large model, so just 10 epochs would be fine).
Optionally, save the trained model for future use.

Model Evaluation
To evaluate the model's performance on test data:

Ensure the test.txt file is correctly formatted.
Use EvaluateModel.java to evaluate the trained model's performance on unseen data.
The evaluation will output key metrics and performance statistics.
Prediction
To make predictions on new images using the trained model:

Ensure the model is either loaded from a saved file or has been recently trained.
Use Predictor.java to predict the nutritional values for a new image.
The predictNutrition method will output the estimated calories, fat, protein, and carbs.
Saving and Loading Models
Saving the Model: After training, the model can be saved using:

model.save(new File("trainedModel.zip"));
Loading the Model: To load a saved model for prediction:

MultiLayerNetwork model = MultiLayerNetwork.load(new File("trainedModel.zip"), true);

Usage
Here is a simple example to get started with the prediction:

public class TestModel {
    public static void main(String[] args) throws IOException {
        // Load the trained model
        MultiLayerNetwork model = MultiLayerNetwork.load(new File("trainedModel.zip"), true);

        // Image path for prediction
        String imagePath = "path_to_image.jpg";

        // Predict nutritional values
        float[] predictedValues = Predictor.predictNutrition(model, imagePath);

        // Output the results
        System.out.println("Predicted Calories: " + predictedValues[0]);
        System.out.println("Predicted Fat: " + predictedValues[1]);
        System.out.println("Predicted Protein: " + predictedValues[2]);
        System.out.println("Predicted Carbs: " + predictedValues[3]);
    }
}
Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue for discussion.

Contact
For any questions or suggestions, feel free to contact p.delalamorodriguez@gmail.com

Happy coding! 😊
