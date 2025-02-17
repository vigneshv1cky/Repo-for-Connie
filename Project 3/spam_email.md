# SMS Spam Collection Dataset Extraction and Analysis

This course demonstrates how to download, extract, analyze, and build a machine learning model using the **SMS Spam Collection** dataset from the UCI Machine Learning Repository. The steps include data extraction, exploratory data analysis (EDA), text preprocessing, model training, evaluation, and saving the final model. This guide is designed for beginners.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installing Conda and Setting Up the Environment](#installing-conda-and-setting-up-the-environment)
3. [Downloading and Extracting the Dataset](#downloading-and-extracting-the-dataset)
4. [Dataset Overview](#dataset-overview)
5. [Visualizing Class Distribution](#visualizing-class-distribution)
6. [Handling Duplicates and Missing Values](#handling-duplicates-and-missing-values)
7. [Converting Labels to Numeric](#converting-labels-to-numeric)
8. [Text Length Distribution Analysis](#text-length-distribution-analysis)
9. [Most Common Words and Word Clouds](#most-common-words-and-word-clouds)
10. [Text Preprocessing and TF-IDF Vectorization](#text-preprocessing-and-tfidf-vectorization)
11. [Splitting Data for Training & Testing](#splitting-data-for-training--testing)
12. [Balancing the Dataset with SMOTE](#balancing-the-dataset-with-smote)
13. [Training and Evaluating the Logistic Regression Model](#training-and-evaluating-the-logistic-regression-model)
14. [Saving and Loading the Final Model](#saving-and-loading-the-final-model)
15. [Conclusion](#conclusion)

---

## Introduction

In this notebook, you'll learn how to:
- Download and extract the SMS Spam Collection dataset.
- Perform exploratory data analysis (EDA).
- Preprocess text data and convert it into numerical features using TF-IDF.
- Balance classes using SMOTE.
- Train and evaluate a Logistic Regression model.
- Save and load the trained model for future predictions.

---

## Installing Conda and Setting Up the Environment

If you don't have **conda** installed, follow these steps:

### Installing Anaconda

1. Visit the [Anaconda Distribution](https://www.anaconda.com/products/distribution) page and download the installer for your operating system.
2. Follow the installation instructions provided on the website.
3. Open your terminal (or Anaconda Prompt on Windows) and verify the installation by running:
   ```bash
   conda --version
   ```

### Setting Up Your Conda Environment

Create and activate a new environment for this project:

1. **Create a new environment (e.g., named `sms_spam`) with Python 3.9:**
   ```bash
   conda create --name sms_spam python=3.10
   ```
2. **Activate the environment:**
   ```bash
   conda activate sms_spam
   ```
3. **Install the required packages:**
   ```bash
   conda install -c conda-forge pandas requests matplotlib seaborn nltk scikit-learn imbalanced-learn xgboost wordcloud numpy scipy
   ```

---

## Downloading and Extracting the Dataset

We begin by downloading the dataset (a ZIP file) and extracting it to a folder.

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

# Load the dataset (the file is tab-separated and has no header)
df = pd.read_csv("spam_dataset/SMSSpamCollection", sep="\t", header=None, names=["label", "text"])

# Display the first few rows
print(df.head())
```

---

## Dataset Overview

Let's check the size and structure of the dataset.

```python
# Check dataset dimensions
print(df.shape)  # Number of rows & columns
print("*******")

# Check data types and missing values
print(df.info())
```

---

## Visualizing Class Distribution

We visualize the distribution of spam and ham messages using a count plot.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Count plot for spam vs. ham messages
sns.countplot(x=df["label"], palette="coolwarm")
plt.title("Spam vs. Ham Distribution")
plt.show()

# Print the counts of each class
print(df["label"].value_counts())
```

---

## Handling Duplicates and Missing Values

Check for duplicate messages and missing values, then clean the dataset if needed.

```python
# Check for duplicate messages
print("Duplicate messages:", df.duplicated().sum())

# Remove duplicate rows if necessary
df = df.drop_duplicates()
print("After removing duplicates:", df.shape)

# Check for missing values
print("Missing values:\n", df.isnull().sum())
```

---

## Converting Labels to Numeric

Convert text labels ("ham", "spam") into numerical values (0 and 1) for modeling.

```python
df["label"] = df["label"].map({"ham": 0, "spam": 1})
print(df["label"].head())
```

---

## Text Length Distribution Analysis

Analyze the distribution of text lengths for both ham and spam messages.

```python
# Create a new column for text length
df["text_length"] = df["text"].apply(len)

# Plot histograms of text length distribution
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

Extract and print the most common words in ham and spam messages.

```python
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string

# Download the stopwords list
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

# Get common words for ham and spam
ham_words = get_most_common_words(df[df["label"] == 0]["text"])
spam_words = get_most_common_words(df[df["label"] == 1]["text"])

print("Most common words in HAM messages:", ham_words)
print("Most common words in SPAM messages:", spam_words)
```

### Word Cloud Generation

Create word clouds to visualize the frequency of words in spam and ham messages.

```python
from wordcloud import WordCloud

# Generate text for word clouds
spam_text = " ".join(df[df["label"] == 1]["text"])
ham_text = " ".join(df[df["label"] == 0]["text"])

plt.figure(figsize=(12, 6))

# Word Cloud for Spam messages
plt.subplot(1, 2, 1)
plt.imshow(WordCloud(width=500, height=300, background_color="white").generate(spam_text))
plt.axis("off")
plt.title("Spam Word Cloud")

# Word Cloud for Ham messages
plt.subplot(1, 2, 2)
plt.imshow(WordCloud(width=500, height=300, background_color="white").generate(ham_text))
plt.axis("off")
plt.title("Ham Word Cloud")

plt.show()
```

---

## Text Preprocessing and TF-IDF Vectorization

Convert the text data into numerical features using TF-IDF vectorization.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF vectorizer (using top 5000 features)
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["text"])  # Convert text to TF-IDF features
y = df["label"]  # Target variable

print(X)
```

---

## Splitting Data for Training & Testing

Split the dataset into training (80%) and testing (20%) sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")
```

Check the class distribution in both sets:

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

Apply SMOTE to the training data to balance the class distribution.

```python
from imblearn.over_sampling import SMOTE

# Initialize SMOTE
smote = SMOTE(sampling_strategy="auto", random_state=42)

# Apply SMOTE to the training data only
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check new class distribution after SMOTE
print("Class distribution after SMOTE:")
print(y_train_resampled.value_counts(normalize=True) * 100)
```

Verify the distribution before and after SMOTE:

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Distribution before SMOTE
sns.barplot(x=y_train.value_counts().index, y=y_train.value_counts(), ax=ax[0])
ax[0].set_title("Before SMOTE")
ax[0].set_xlabel("Class")
ax[0].set_ylabel("Count")

# Distribution after SMOTE
sns.barplot(x=y_train_resampled.value_counts().index, y=y_train_resampled.value_counts(), ax=ax[1])
ax[1].set_title("After SMOTE")
ax[1].set_xlabel("Class")
ax[1].set_ylabel("Count")

plt.show()
```

---

## Training and Evaluating the Logistic Regression Model

Train a Logistic Regression model on the balanced dataset and evaluate its performance.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Train the Logistic Regression model on the SMOTE-resampled data
model_smote = LogisticRegression()
model_smote.fit(X_train_resampled, y_train_resampled)

# Predict on the original test set
y_pred_smote = model_smote.predict(X_test)

# Print evaluation metrics
print("Classification Report after SMOTE:")
print(classification_report(y_test, y_pred_smote))

# Generate and plot the confusion matrix
conf_matrix_smote = confusion_matrix(y_test, y_pred_smote)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_smote, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix after SMOTE")
plt.show()
```

---

## Saving and Loading the Final Model

Save the trained model and TF-IDF vectorizer for future use. You can later load them to make predictions on new messages.

```python
import joblib

# Save the model and vectorizer to disk
joblib.dump(model_smote, "best_spam_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")
```

Test loading the saved model and making a prediction:

```python
# Load the saved model and vectorizer
loaded_model = joblib.load("best_spam_classifier_model.pkl")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Test the saved model with a new message
new_message = ["Congratulations, you've won a $1000 gift card! Click here to claim."]
new_message_transformed = loaded_vectorizer.transform(new_message)
prediction = loaded_model.predict(new_message_transformed)

print(f"The new message is classified as: {'Spam' if prediction[0] == 1 else 'Ham'}")
```

---

## Conclusion

In this course, you learned how to:
- Download and extract the SMS Spam Collection dataset.
- Perform exploratory data analysis and data cleaning.
- Preprocess and vectorize text data using TF-IDF.
- Balance the dataset using SMOTE.
- Train and evaluate a Logistic Regression model.
- Save and load the trained model for future predictions.

This comprehensive guide provides a solid foundation for beginners in machine learning and natural language processing. Happy coding!
