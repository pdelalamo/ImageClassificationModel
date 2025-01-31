# Food Image Nutritional Information Estimator

## Project Overview

The **Food Image Nutritional Information Estimator** is a deep learning model designed to predict the nutritional content of food items based on images. The model outputs four key nutritional values:

- **Calories**
- **Fat**
- **Protein**
- **Carbohydrates**

The project uses a **Convolutional Neural Network (CNN)** architecture, implemented with the **Deeplearning4j** library. This includes steps for data preprocessing, model training, evaluation, and prediction.

## Project Structure

The project structure is as follows:

com.fitmymacros.imageclassifiermodel/ │ ├── CNNModel.java # Defines and trains the CNN model ├── DataParser.java # Parses metadata and image paths for training/testing ├── DataSetUtility.java # Converts data to DataSetIterator for training ├── EvaluateModel.java # Evaluates the model's performance on test data ├── ImageLoader.java # Loads and preprocesses images ├── Predictor.java # Makes predictions based on the trained model └── TestModel.java # Tests the model with new images


## Requirements

To run the project, you will need the following dependencies:

- **Java 8+**
- **Deeplearning4j 1.0.0-beta7**
- **Nd4j 1.0.0-beta7**
- **Datavec 1.0.0-beta7**

Install the necessary dependencies using your preferred package management tool (e.g., Maven or Gradle).

## Installation

Follow these steps to set up the project:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/pdelalamo/ImageClassificationModel.git
   cd ImageClassificationModel
   ```

2. **Configure the Project:**

Update the paths in DataParser.java and TestModel.java to point to the correct directories for images and metadata on your local system.
Build the Project:

Build the project using your preferred build tool (e.g., Maven):

```bash
mvn clean install
```

3. **Usage**
4. 
After building the project, you can use it for the following tasks:

- Training the Model: Run CNNModel.java to train the model using your image dataset.
- Evaluating the Model: Use EvaluateModel.java to evaluate the model's performance on the test dataset.
- Making Predictions: Run Predictor.java to make nutritional predictions based on new food images.

**Contributing**

I welcome contributions to this project. If you'd like to contribute, please fork the repository, make your changes, and submit a pull request.
