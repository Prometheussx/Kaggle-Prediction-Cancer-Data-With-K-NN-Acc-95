
# Cancer Data Classification with K-Nearest Neighbors (K-NN)

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Data Preparation](#data-preparation)
6. [Usage](#usage)
    - [Training the Model](#training-the-model)
    - [Evaluation](#evaluation)
7. [Results](#results)
8. [License](#license)
9. [Author](#author)

## Overview

This project employs the K-Nearest Neighbors (K-NN) algorithm to classify cancer cells as either benign or malignant. With an impressive accuracy rate of 96.50%, this model can be used for medical diagnosis and research purposes.

## Dataset

The project uses the [Cancer_Data.csv](https://www.kaggle.com/datasets/erdemtaha/cancer-data) dataset. This dataset contains 569 entries with 30 features, and each entry is labeled as benign (B) or malignant (M).

## Project Structure

The project is organized into the following sections:

1. **Library and Input File:** This section imports necessary libraries and loads the dataset.

2. **Data Loading and Editing:** The dataset is loaded, and unnecessary columns (such as 'Unnamed: 32' and 'id') are removed. The 'diagnosis' column is also converted to numerical values (1 for 'M' and 0 for 'B').

3. **Normalization:** Data normalization is performed to scale the values between 0 and 1, preventing high or low values from introducing errors in the model.

4. **Train Test Split:** The dataset is divided into training and testing sets for model training and evaluation.

5. **Initialize K-NN Classifier:** The K-NN classifier is initialized with a specified number of neighbors (k).

6. **Model Training:** The K-NN model is trained on the training data.

7. **Model Evaluation:** The model's performance is evaluated, including a confusion matrix.

8. **Model Result:** The training progress is monitored, and the model's performance is assessed.

## Requirements

To run the project, make sure you have the following Python libraries installed:

- NumPy
- pandas
- scikit-learn
- seaborn
- matplotlib

You can install these libraries using pip:

```bash
pip install numpy
pip install pandas
pip install scikit-learn
pip install seaborn
pip install matplotlib
```

## Getting Started

### Installation

1. Clone the project repository:

```bash
git clone https://github.com/Prometheussx/Kaggle-Notebook-Cancer-Prediction-ACC96.5-With-K-NN.git
cd Kaggle-Notebook-Cancer-Prediction-ACC96.5-With-K-NN
```

2. Ensure you have Python and the required libraries installed.

### Data Preparation

1. Download the [Cancer_Data.csv](https://www.kaggle.com/datasets/erdemtaha/cancer-data) dataset and place it in the project directory.

2. Follow the code in the "Data Loading and Editing" section to load and preprocess the dataset.

## Usage

### Training the Model

Execute the Python code in the repository files to perform K-NN classification and train the model.

```python
Example: knn_classification(x_train, y_train, x_test, y_test, k=5)
```

### Evaluation

The model's performance is evaluated with metrics such as accuracy and a confusion matrix.

## Results

The project attains an accuracy of 96.50% in classifying cancer cells. Training progress and results are visualized in the README.

## License

This project is released under the [MIT License](https://github.com/Prometheussx/Kaggle-Notebook-Cancer-Prediction-ACC96.5-With-K-NN/blob/main/LICENSE).

## Author
- Email Address: [Erdem Taha Sokullu](mailto:erdemtahasokullu@gmail.com)
- LinkedIn Profile: [Erdem Taha Sokullu](https://www.linkedin.com/in/erdem-taha-sokullu/)
- GitHub Profile: [Prometheussx](https://github.com/Prometheussx)
- Kaggle Profile: [@erdemtaha](https://www.kaggle.com/erdemtaha)

**Feel free to reach out if you have any questions or need further information about the project.**
