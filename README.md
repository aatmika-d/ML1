# ML1

## Project Notebooks Overview

### 2. `kmeansclustering.ipynb`
- Applies **K-Means clustering** on the cleaned dataset.
- Finds groups (clusters) of similar customers.
- Visualizes clusters to understand distinct customer segments.
- Analyzes characteristics of each cluster (e.g., high usage vs low usage groups).

### 3. `knnclassification.ipynb`
- Builds a **K-Nearest Neighbors (KNN)** classifier.
- Trains KNN to predict customer churn.
- Evaluates model performance using accuracy and other metrics.

### 4. `logreg.ipynb`
- Builds a **Logistic Regression** model for churn prediction.
- Trains the model on training data.
- Tests the model and reports accuracy, precision, recall, and F1-score.
- Interprets model coefficients to understand feature impact on churn.

### 5. `decisiontree.ipynb`
- Builds a **Decision Tree** model for churn prediction.
- Trains and evaluates the decision tree.
- Visualizes the tree structure to explain decision paths for churn prediction.

### 7. `pca.ipynb`
- Performs **Principal Component Analysis (PCA)** for dimensionality reduction.
- Identifies the most important features.
- Visualizes PCA components.
- Helps simplify data and improve model efficiency.

### 8. `wrappermethods.ipynb`
- Applies **Wrapper feature selection methods**, such as Sequential Feature Selector.
- Finds the best subset of features to improve model accuracy.
- Uses models to iteratively evaluate different feature combinations.

### 9. `filtering.ipynb`
- Uses **Filter feature selection techniques** based on statistical tests.
- Selects features with strong correlation to churn.
- Quickly filters out irrelevant or noisy features.

### 10. `test.ipynb`
- Runs tests and validation on models and selected features.
- Checks model performance on unseen data.
- Compares different models or feature sets for best results.

---

## Streamlit App (`app.py`)

- A user-friendly web application built using **Streamlit**.
- Allows users to input customer data and choose between:
  - Logistic Regression for churn prediction,
  - KNN classification for churn prediction,
  - K-Means clustering for customer segmentation.
- Provides predictions and cluster assignments instantly.
- Designed to showcase the models developed in the notebooks.
