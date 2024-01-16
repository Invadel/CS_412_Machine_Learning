# README for Predicting Student Grades Based on ChatGPT Interactions

This repository contains the code and methodology for a machine learning project aimed at predicting student homework grades from their interactions with ChatGPT. The dataset comprises ChatGPT history in HTML format and corresponding CSV files with student scores.

## Project Overview

### Data Description

- **ChatGPT Histories**: Collection of 150-200 HTML files containing student interactions with ChatGPT.
- **Student Scores**: A CSV file with the final grades of students.

### Objective

The project's goal is to predict the points students received for individual homework questions and their total grade using a two-step machine learning pipeline that involves clustering and a neural network.

## Methodology

### Preprocessing

1. **Cosine Similarity Scoring**: Calculated between user prompts and homework questions to obtain similarity scores for each question.
2. **Dataset Splitting with Stratification**: Attention was given to 11 rare classes in the dataset, which were first mapped to fewer categories before splitting.
3. **Dataframe Creation**: Separate dataframes for training (`x_train`, `y_train`) and testing (`x_test`, `y_test`) were prepared.

### Clustering Algorithm for Estimating Points

#### Rationale for Clustering

Due to the limited and complex nature of the dataset, it was challenging to derive features and relationships directly from the ChatGPT interactions. We lacked specific labels for the points received from each homework question. To address this, we employed an unsupervised technique, K-means clustering, to estimate these points.

#### Implementation

- **K-Means Clustering**: Applied to each question column to assign cluster labels to each student's response.
- **Feature Formulation**: Cluster centers for each question were used to formulate new features, including the points received per question and a total calculated grade.
- **Insight Enhancement**: This approach significantly enhanced the dataset's insights, enabling the subsequent application of a neural network model for total grade prediction.

### Neural Network Model for Grade Prediction

#### Inputs

- Includes metrics like user prompts, error counts, entropy, average characters in prompts and responses, points received per question, and the calculated total grade.

#### Architecture

- Custom neural network with residual blocks, designed for regression tasks.

#### Training and Evaluation

- Trained over 200 epochs with mean squared error loss. Performance evaluated using accuracy, precision, and recall metrics.

#### Testing and Performance Evaluation

- **Testing Dataset Preparation**: The testing dataset was also clustered using the cluster labels and centers learned from the training dataset.
- **Model Testing**: The neural network model was then tested with this prepared testing dataset.
- **Evaluation**: The performance of the model was evaluated based on how well it predicted the total grades.

## Neural Network Code

The neural network is implemented using PyTorch. It includes data preparation, model definition, training, and evaluation steps, with a focus on performance metrics and regression plot visualization.

## Usage

1. **Data Preparation**: Format your data as described in the Data Description section.
2. **Model Training**: Run the neural network code to train the model on your data.
3. **Model Evaluation**: Evaluate the model's performance using the provided metrics.
4. **Prediction**: Use the trained model to predict student grades based on new ChatGPT interaction data.

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Contributors

This project is open for contributions. Please read the contribution guidelines before

submitting a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

### Detailed Explanation of Approaches and Algorithms

#### Motivation Behind Clustering Approach

The clustering approach was crucial due to the unique challenges posed by the dataset:
- **Limited Data**: The dataset's limited scope made it difficult to extract meaningful features directly.
- **Complexity of ChatGPT Interactions**: The intricate nature of conversational data added to the complexity.
- **Absence of Specific Labels**: We lacked detailed labels for individual question scores.

K-means clustering was chosen for its efficiency in grouping similar data points, thereby enabling us to estimate scores for each question based on the cluster characteristics.

#### Benefits of the Clustering Approach

- **Feature Enhancement**: It allowed us to create meaningful features from an otherwise limited dataset.
- **Unsupervised Learning Advantage**: As an unsupervised technique, it bypassed the need for labeled data for individual question scores.
- **Foundation for Neural Network**: The features derived from clustering (cluster centers, estimated points) provided a solid base for the neural network model.

#### Neural Network Model: A Step Forward

After clustering, the neural network model capitalized on the newly formulated features to predict total grades. This two-step approach synergized unsupervised and supervised learning techniques, utilizing the strengths of each to overcome data limitations.

#### Testing and Validation

- **Cluster-Based Testing**: By applying the same clustering labels and centers to the testing dataset, we ensured consistency and relevance in testing.
- **Comprehensive Model Evaluation**: The model's success was measured not just in its ability to predict but also in how it generalized to unseen data.

### Conclusion and Future Work

This project exemplifies the effective combination of clustering and neural network techniques in a challenging data environment. Future work could explore more complex clustering algorithms, deeper neural network architectures, or alternative supervised learning methods to enhance prediction accuracy and model robustness.

---

*This README provides a comprehensive guide to understanding the methodologies, rationales, and technical details of the project. It serves as both a documentation and a starting point for those interested in exploring the intersection of machine learning and educational data analysis.*
