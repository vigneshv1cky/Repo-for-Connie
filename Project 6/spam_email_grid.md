


# SMS Spam Collection Dataset Extraction and Analysis

This course is designed for beginners. In this notebook, we will download, extract, and analyze the **SMS Spam Collection** dataset from the UCI Machine Learning Repository. We cover topics such as data extraction, preprocessing, exploratory analysis, model training, hyperparameter tuning, and saving the final model for future use.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installing Conda and Setting Up the Environment](#installing-conda-and-setting-up-the-environment)
3. [Downloading and Extracting the Dataset](#downloading-and-extracting-the-dataset)
4. [Dataset Overview and Analysis](#dataset-overview-and-analysis)
5. [Visualizing Class Distribution](#visualizing-class-distribution)
6. [Handling Duplicates and Missing Values](#handling-duplicates-and-missing-values)
7. [Converting Labels to Numeric](#converting-labels-to-numeric)
8. [Text Length Distribution Analysis](#text-length-distribution-analysis)
9. [Most Common Words and Word Clouds](#most-common-words-and-word-clouds)
10. [Text Preprocessing and Vectorization](#text-preprocessing-and-vectorization)
11. [Splitting Data for Training & Testing](#splitting-data-for-training--testing)
12. [Balancing the Dataset with SMOTE](#balancing-the-dataset-with-smote)
13. [Training a Logistic Regression Model](#training-a-logistic-regression-model)
14. [Evaluating the Model](#evaluating-the-model)
15. [Hyperparameter Tuning](#hyperparameter-tuning)
16. [Saving and Loading the Final Model](#saving-and-loading-the-final-model)
17. [Conclusion](#conclusion)

---

## Introduction

In this course, you will learn how to work with a real-world SMS spam dataset. We will cover:

- Downloading and extracting data from the UCI repository.
- Exploratory data analysis (EDA) to understand dataset characteristics.
- Preprocessing text data and converting it into numerical features.
- Balancing classes using SMOTE.
- Training and evaluating a Logistic Regression model.
- Hyperparameter tuning using GridSearchCV and RandomizedSearchCV.
- Saving and loading the trained model for future predictions.

---

## Installing Conda and Setting Up the Environment

### Installing Anaconda

1. Visit the [Anaconda Distribution](https://www.anaconda.com/products/distribution) page and download the installer for your operating system.
2. Follow the installation instructions provided on the website.
3. Open your terminal (or Anaconda Prompt on Windows) and verify the installation by running:

   ```bash
   conda --version
   ```

### Setting Up Your Conda Environment

After installing conda, create and activate a new environment for this project:

```bash
conda create --name sms_spam python=3.10
conda activate sms_spam
conda install -c conda-forge pandas requests matplotlib seaborn nltk scikit-learn imbalanced-learn xgboost wordcloud numpy scipy
```

---

## Downloading and Extracting the Dataset

The following code downloads the SMS Spam Collection dataset and extracts it into a folder:

```python
import pandas as pd
import requests
import zipfile
import io

# URL of the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

# Download the ZIP file
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_ref:
    zip_ref.extractall("spam_dataset")  # Extract files to a folder

# Load the dataset (the file is tab-separated)
df = pd.read_csv("spam_dataset/SMSSpamCollection", sep="\t", header=None, names=["label", "text"])

# Display the first few rows of the dataset
print(df.head())
```

---

## Dataset Overview and Analysis

Check the size and structure of the dataset:

```python
# Dataset dimensions (rows and columns)
print(df.shape)
print("*******")
# Data types and missing values
print(df.info())
```

---

## Visualizing Class Distribution

Visualize the distribution of spam and ham messages:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Count plot of spam vs. ham messages
sns.countplot(x=df["label"], palette="coolwarm")
plt.title("Spam vs. Ham Distribution")
plt.show()

# Print the counts of each class
print(df["label"].value_counts())
```

---

## Handling Duplicates and Missing Values

Check for duplicate messages and missing values:

```python
# Check for duplicate rows
print("Duplicate messages:", df.duplicated().sum())

# Remove duplicates if necessary
df = df.drop_duplicates()
print("After removing duplicates:", df.shape)

# Check for missing values in the dataset
print("Missing values:\n", df.isnull().sum())
```

---

## Converting Labels to Numeric

Convert textual labels to numeric values (ham=0, spam=1):

```python
df["label"] = df["label"].map({"ham": 0, "spam": 1})
print(df["label"])
```

---

## Text Length Distribution Analysis

Analyze and visualize the distribution of text lengths in the messages:

```python
# Create a new column for text length
df["text_length"] = df["text"].apply(len)

# Plot histogram for text length distribution in ham and spam messages
plt.figure(figsize=(10, 5))
sns.histplot(df[df["label"] == 0]["text_length"], bins=30, kde=True, color="blue")
sns.histplot(df[df["label"] == 1]["text_length"], bins=30, kde=True, color="red")
plt.legend(labels=["Ham", "Spam"])
plt.title("Text Length Distribution (Spam vs. Ham)")
plt.show()
```

---

## Most Common Words and Word Clouds

### Most Common Words

Extract and display the most common words in ham and spam messages:

```python
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords if not already downloaded
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def get_most_common_words(texts, n=20):
    words = (
        " ".join(texts)
        .lower()
        .translate(str.maketrans("", "", string.punctuation))
        .split()
    )
    words = [word for word in words if word not in stop_words]
    return Counter(words).most_common(n)

# Get most common words for ham and spam
ham_words = get_most_common_words(df[df["label"] == 0]["text"])
spam_words = get_most_common_words(df[df["label"] == 1]["text"])

print("Most common words in HAM messages:", ham_words)
print("Most common words in SPAM messages:", spam_words)
```

### Word Cloud Generation

Generate word clouds to visually display the most frequent words:

```python
from wordcloud import WordCloud

# Generate text for spam and ham messages
spam_text = " ".join(df[df["label"] == 1]["text"])
ham_text = " ".join(df[df["label"] == 0]["text"])

plt.figure(figsize=(12, 6))

# Word Cloud for Spam Messages
plt.subplot(1, 2, 1)
plt.imshow(WordCloud(width=500, height=300, background_color="white").generate(spam_text))
plt.axis("off")
plt.title("Spam Word Cloud")

# Word Cloud for Ham Messages
plt.subplot(1, 2, 2)
plt.imshow(WordCloud(width=500, height=300, background_color="white").generate(ham_text))
plt.axis("off")
plt.title("Ham Word Cloud")

plt.show()
```

---

## Text Preprocessing and Vectorization

Convert the text data into numerical features using TF-IDF vectorization:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the vectorizer (using top 5000 features and English stop words)
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["text"])  # Transform text into TF-IDF features
y = df["label"]

print(X)
```

---

## Splitting Data for Training & Testing

Split the dataset into training (80%) and testing (20%) sets:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")
```

Check the class distribution in the training and testing sets:

```python
train_distribution = y_train.value_counts(normalize=True) * 100
test_distribution = y_test.value_counts(normalize=True) * 100

print("Class distribution in Training Set:")
print(train_distribution)

print("\nClass distribution in Testing Set:")
print(test_distribution)
```

---

## Balancing the Dataset with SMOTE

Use SMOTE to balance the classes in the training set:

```python
from imblearn.over_sampling import SMOTE

# Initialize SMOTE
smote = SMOTE(sampling_strategy="auto", random_state=42)

# Apply SMOTE on the training data only
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check new class distribution after SMOTE
print("Class distribution after SMOTE:")
print(y_train_resampled.value_counts(normalize=True) * 100)
```

Visualize the class distribution before and after applying SMOTE:

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot distribution before SMOTE
sns.barplot(x=y_train.value_counts().index, y=y_train.value_counts(), ax=ax[0])
ax[0].set_title("Before SMOTE")
ax[0].set_xlabel("Class")
ax[0].set_ylabel("Count")

# Plot distribution after SMOTE
sns.barplot(x=y_train_resampled.value_counts().index, y=y_train_resampled.value_counts(), ax=ax[1])
ax[1].set_title("After SMOTE")
ax[1].set_xlabel("Class")
ax[1].set_ylabel("Count")

plt.show()
```

---

## Training a Logistic Regression Model

Train a Logistic Regression model using the balanced dataset:

```python
from sklearn.linear_model import LogisticRegression

# Train the logistic regression model on the SMOTE-balanced data
model_smote = LogisticRegression()
model_smote.fit(X_train_resampled, y_train_resampled)

# Predict on the original test set
y_pred_smote = model_smote.predict(X_test)
```

---

## Evaluating the Model

Evaluate the model using classification reports and a confusion matrix:

```python
from sklearn.metrics import classification_report, confusion_matrix

# Print the classification report
print("Classification Report after SMOTE:")
print(classification_report(y_test, y_pred_smote))

# Create a confusion matrix
conf_matrix_smote = confusion_matrix(y_test, y_pred_smote)

# Visualize the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_smote, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix after SMOTE")
plt.show()
```

---

## Hyperparameter Tuning

### 1. GridSearchCV for Regularization Tuning

Use GridSearchCV to find the best regularization parameter \( C \):

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "C": [0.01, 0.1, 1, 10, 50, 100, 250, 500, 1000]
}

grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring="roc_auc")
grid_search.fit(X_train_resampled, y_train_resampled)

best_model = grid_search.best_estimator_
print(f"Best Regularization Parameter: {grid_search.best_params_['C']}")
```

### 2. RandomizedSearchCV for ElasticNet Regularization

Use RandomizedSearchCV for faster hyperparameter tuning with ElasticNet:

```python
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

param_dist = {
    "C": np.logspace(-3, 3, 20),  # Logarithmic scale sampling
    "penalty": ["elasticnet"],
    "l1_ratio": np.linspace(0, 1, 20),
}

random_search = RandomizedSearchCV(
    LogisticRegression(max_iter=1000, solver="saga"),
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42,
)

random_search.fit(X_train_resampled, y_train_resampled)

print(f"Best Regularization Parameter (C) from RandomizedSearchCV: {random_search.best_params_['C']}")
print(f"Best L1 Ratio from RandomizedSearchCV: {random_search.best_params_['l1_ratio']}")
```

### 3. Training with the Best Hyperparameters

Train the final Logistic Regression model with the tuned hyperparameters:

```python
model_best = LogisticRegression(
    C=random_search.best_params_['C'],
    penalty="elasticnet",
    l1_ratio=0.0,  # Adjust the L1 ratio if needed
    solver="saga",
    max_iter=1000
)
model_best.fit(X_train_resampled, y_train_resampled)
y_pred_smote = model_best.predict(X_test)
```

Evaluate the updated model:

```python
print("Classification Report after SMOTE:")
print(classification_report(y_test, y_pred_smote))

conf_matrix_smote = confusion_matrix(y_test, y_pred_smote)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_smote, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix after SMOTE")
plt.show()
```

---

## Saving and Loading the Final Model

Save the trained model and the TF-IDF vectorizer for future predictions:

```python
import joblib

# Save the best model and vectorizer
joblib.dump(model_smote, "best_spam_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")
```

You can later load them to make predictions on new messages:

```python
# Load the saved model and vectorizer
loaded_model = joblib.load("best_spam_classifier_model.pkl")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Test with a new message
new_message = ["Congratulations, you've won a $1000 gift card! Click here to claim."]
new_message_transformed = loaded_vectorizer.transform(new_message)
prediction = loaded_model.predict(new_message_transformed)

print(f"The new message is classified as: {'Spam' if prediction[0] == 1 else 'Ham'}")
```

---

## Conclusion

In this course, you learned how to:

- Download and extract the SMS Spam Collection dataset.
- Perform exploratory data analysis to understand data characteristics.
- Preprocess and vectorize text data using TF-IDF.
- Balance the dataset using SMOTE.
- Train and evaluate a Logistic Regression model.
- Tune hyperparameters with GridSearchCV and RandomizedSearchCV.
- Save and load the trained model for future predictions.

This comprehensive guide provides a solid foundation for beginners in machine learning and natural language processing. Happy coding!
