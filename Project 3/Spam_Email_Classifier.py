import matplotlib.pyplot as plt

plt.style.use("ggplot")

###############################################
# Download & Extract the ZIP File
###############################################

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

# Load the dataset (it's inside the extracted folder)
df = pd.read_csv(
    "spam_dataset/SMSSpamCollection", sep="\t", header=None, names=["label", "text"]
)

# Display first few rows
print(df.head())

###############################################
# Check Dataset Overview
###############################################

# Check dataset size and structure
print(df.shape)  # Rows & Columns
print("*******")
print(df.info())  # Data types & Missing values

###############################################
# Check Class Distribution (Spam vs. Ham)
###############################################

import matplotlib.pyplot as plt
import seaborn as sns

# Count plot of spam vs ham messages
sns.countplot(x=df["label"], palette="coolwarm")
plt.title("Spam vs. Ham Distribution")
plt.show()

# Print counts
print(df["label"].value_counts())

###############################################
# Checking for Duplicate Messages/Rows
###############################################

print("Duplicate messages:", df.duplicated().sum())

# Remove duplicates if necessary
df = df.drop_duplicates()
print("After removing duplicates:", df.shape)

###############################################
# Checking for Missing Rows
###############################################

print("Missing values:\n", df.isnull().sum())

###############################################
# Convert Labels to Numeric (Machine learning models require numerical labels.)
###############################################

df["label"] = df["label"].map({"ham": 0, "spam": 1})

###############################################
# Check Text Length Distribution
###############################################

df["text_length"] = df["text"].apply(len)

# Histogram of text length distribution

plt.figure(figsize=(10, 5))
sns.histplot(df[df["label"] == 0]["text_length"], bins=30, kde=True, color="blue")
sns.histplot(df[df["label"] == 1]["text_length"], bins=30, kde=True, color="red")
plt.legend(labels=["Ham", "Spam"])  # Explicitly provide the labels
plt.title("Text Length Distribution (Spam vs. Ham)")
plt.show()

###############################################
#  Most Common Words in Spam vs. Ham
###############################################

from collections import Counter
import nltk
from nltk.corpus import stopwords
import string

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


# Function to get common words
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


###############################################
# Word Cloud for Spam and Ham Messages
###############################################

from wordcloud import WordCloud

# Generate word clouds
spam_text = " ".join(df[df["label"] == 1]["text"])
ham_text = " ".join(df[df["label"] == 0]["text"])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(
    WordCloud(width=500, height=300, background_color="white").generate(spam_text)
)
plt.axis("off")
plt.title("Spam Word Cloud")

plt.subplot(1, 2, 2)
plt.imshow(
    WordCloud(width=500, height=300, background_color="white").generate(ham_text)
)
plt.axis("off")
plt.title("Ham Word Cloud")

plt.show()


###############################################
# Text Preprocessing and Vectorization
###############################################

"""

Before training the model, we need to:

1. Convert text data into numerical format using TF-IDF Vectorization.
2. Split the dataset into training and testing sets.

"""

from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text data to numerical representation using TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english", max_features=5000
)  # Use top 5000 words
X = vectorizer.fit_transform(df["text"])  # Transform text into TF-IDF features
y = df["label"]  # Target variable (spam=1, ham=0)

# get indexing
print("\nWord indexes:")
print(vectorizer.vocabulary_)

# display tf-idf values
print("\ntf-idf values:")
print(X)

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Convert to DataFrame with words as column names
df_tfidf = pd.DataFrame(X.toarray(), columns=feature_names)

# Print the TF-IDF matrix
print(df_tfidf)

###############################################
# Split Data for Training & Testing
###############################################

from sklearn.model_selection import train_test_split

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")


###############################################
# Check Class Distribution After Splitting
###############################################

# Check distribution in training and testing sets
train_distribution = y_train.value_counts(normalize=True) * 100  # Percentage format
test_distribution = y_test.value_counts(normalize=True) * 100  # Percentage format

print("Class distribution in Training Set:")
print(train_distribution)

print("\nClass distribution in Testing Set:")
print(test_distribution)


###############################################
# Apply SMOTE for Oversampling
###############################################

"""

SMOTE generates synthetic samples for the minority class (spam), making the dataset balanced.
sampling_strategy='auto' balances both classes equally.
It is only applied to the training set to prevent data leakage.

"""

from imblearn.over_sampling import SMOTE

# Initialize SMOTE
smote = SMOTE(sampling_strategy="auto", random_state=42)

# Apply SMOTE only on training data (to avoid data leakage)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check new class distribution
print("Class distribution after SMOTE:")
print(y_train_resampled.value_counts(normalize=True) * 100)

###############################################
# Verify Class Distribution
###############################################

import matplotlib.pyplot as plt
import seaborn as sns

# Plot class distribution before and after SMOTE
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Before SMOTE
sns.barplot(x=y_train.value_counts().index, y=y_train.value_counts(), ax=ax[0])
ax[0].set_title("Before SMOTE")
ax[0].set_xlabel("Class")
ax[0].set_ylabel("Count")

# After SMOTE
sns.barplot(
    x=y_train_resampled.value_counts().index,
    y=y_train_resampled.value_counts(),
    ax=ax[1],
)
ax[1].set_title("After SMOTE")
ax[1].set_xlabel("Class")
ax[1].set_ylabel("Count")

plt.show()

###############################################
# Train Logistic Regression Model on Balanced Data
###############################################

from sklearn.linear_model import LogisticRegression

# Train logistic regression model on the balanced dataset
model_smote = LogisticRegression()
model_smote.fit(X_train_resampled, y_train_resampled)

# Predict on the original test set
y_pred_smote = model_smote.predict(X_test)

###############################################
# Evaluate the Model
###############################################

from sklearn.metrics import classification_report, confusion_matrix

# Print evaluation metrics
print("Classification Report after SMOTE:")
print(classification_report(y_test, y_pred_smote))

# Confusion matrix
conf_matrix_smote = confusion_matrix(y_test, y_pred_smote)

# Visualizing confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(
    conf_matrix_smote,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix after SMOTE")
plt.show()

##############################################################################################
# Hyperparameter Tuning - Logistic Regression
###############################################

# Import necessary libraries
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
import time
import numpy as np

# Section 1: Basic GridSearchCV - Regularization Tuning (Initial Step)
###############################################

# Define hyperparameters to tune
param_grid = {
    "C": [0.01, 0.1, 1, 10, 50, 100, 250, 500, 1000]  # Regularization strength
}

# Use GridSearchCV for best parameter selection with 5-fold cross-validation
grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000), param_grid, cv=5, scoring="roc_auc"
)

# Fit the grid search to the data
grid_search.fit(X_train_resampled, y_train_resampled)

# Extract and print the best model and the best regularization parameter
best_model = grid_search.best_estimator_
print(f"Best Regularization Parameter: {grid_search.best_params_['C']}")


###############################################
# Improve Model Performance
# Hyperparameter Tuning with RandomizedSearchCV
###############################################

from scipy.stats import loguniform

# Define the hyperparameter search space
param_dist = {
    "C": loguniform(0.001, 100)
}  # Using loguniform for better search in large ranges

# Use RandomizedSearchCV for best parameter selection
random_search = RandomizedSearchCV(
    LogisticRegression(max_iter=1000),
    param_distributions=param_dist,
    n_iter=20,  # Number of random samples
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
)

random_search.fit(X_train_resampled, y_train_resampled)

# Best model
best_model = random_search.best_estimator_
print(f"Best Regularization Parameter: {random_search.best_params_['C']}")


###############################################
# Hyperparameter Tuning - RandomizedSearchCV
###############################################

# Section 3: RandomizedSearchCV for ElasticNet Regularization
# Define the hyperparameter distribution for RandomizedSearchCV
param_dist = {
    "C": np.logspace(-3, 3, 20),  # Log scale for wide range sampling
    "penalty": ["elasticnet"],  # ElasticNet penalty
    "l1_ratio": np.linspace(0, 1, 20),  # Randomly sample ratio between 0 and 1
}

# Use RandomizedSearchCV for a faster random sampling over the hyperparameter space
# n_iter defines how many random combinations to sample
random_search = RandomizedSearchCV(
    LogisticRegression(max_iter=100, solver="saga"),
    param_distributions=param_dist,
    n_iter=10,  # Number of random combinations to try
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,  # Use all available CPU cores
    random_state=42,  # For reproducibility
)

# Fit the RandomizedSearchCV model
random_search.fit(X_train_resampled, y_train_resampled)

# Extract and print the best model and parameters from RandomizedSearchCV
print(
    f"Best Regularization Parameter (C) from RandomizedSearchCV: {random_search.best_params_['C']}"
)
print(
    f"Best L1 Ratio from RandomizedSearchCV: {random_search.best_params_['l1_ratio']}"
)


###############################################
# Hyperparameter Tuning - ElasticNet Regularization
###############################################

"""

# Section 2: GridSearchCV with ElasticNet Regularization
# Using ElasticNet to tune both `C` and `l1_ratio` hyperparameters
param_grid = {
    "C": [0.01, 0.1, 1, 10, 50, 100],  # Regularization strength (reduced)
    "penalty": ["elasticnet"],  # ElasticNet penalty
    "l1_ratio": [0.1, 0.5, 0.9],  # ElasticNet ratio between L1 and L2 (reduced)
}

# Start the timer to measure execution time
start_time = time.time()

# GridSearchCV for ElasticNet tuning with parallelism (n_jobs=-1) for faster computation
grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, solver="saga"),
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,  # Use all available CPU cores for parallelization
)

# Fit the model to the data
grid_search.fit(X_train_resampled, y_train_resampled)

# Calculate and print the time taken for the grid search
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")

# Extract and print the best model and parameters from GridSearchCV
best_model = grid_search.best_estimator_
print(f"Best Regularization Parameter (C): {grid_search.best_params_['C']}")
print(f"Best L1 Ratio: {grid_search.best_params_['l1_ratio']}")

"""

###############################################
# Hyperparameter Tuning for -
# LogisticRegression,
# RandomForest,
# SVC,
# KNeighborsClassifier,
# AdaBoostClassifier
# GradientBoostingClassifier
###############################################

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import numpy as np


# Define hyperparameter distributions for different models
param_grids = {
    "LogisticRegression": {
        "C": np.logspace(-3, 3, 20),
        "penalty": ["elasticnet"],
        "l1_ratio": np.linspace(0, 1, 20),
        "solver": ["saga"],
    },
    "RandomForest": {
        "n_estimators": np.arange(50, 500, 50),
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    },
    "SVC": {
        "C": np.logspace(-3, 3, 20),
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
        "gamma": ["scale", "auto"],
        "degree": np.arange(1, 5),
    },
    "KNeighborsClassifier": {
        "n_neighbors": np.arange(3, 30, 2),
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": np.arange(20, 100, 10),
        "p": [1, 2],  # 1 for Manhattan distance, 2 for Euclidean distance
    },
    "AdaBoostClassifier": {
        "n_estimators": np.arange(50, 500, 50),
        "learning_rate": [0.001, 0.01, 0.1, 1],
    },
    "GradientBoostingClassifier": {
        "n_estimators": np.arange(50, 500, 50),
        "learning_rate": [0.001, 0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.8, 0.9, 1.0],
    },
}

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=100, solver="saga"),
    "RandomForest": RandomForestClassifier(),
    "SVC": SVC(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
}


# Perform RandomizedSearchCV for each model
best_params = {}
best_recall_scores = {}

# Initialize a dictionary to store the recall scores for each model and fold
fold_recall_scores = {model_name: [] for model_name in models}

###############################################
# Perform RandomizedSearchCV for each model
###############################################

for model_name, model in models.items():
    print(f"Running RandomizedSearchCV for {model_name}...")

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grids[model_name],
        n_iter=10,  # Number of random combinations to try
        cv=5,
        scoring="recall",
        n_jobs=-1,
        random_state=42,
        return_train_score=False,  # We don't need training scores
    )

    random_search.fit(X_train_resampled, y_train_resampled)
    best_params[model_name] = random_search.best_params_
    best_recall_scores[model_name] = random_search.best_score_

    # Capture the recall score for each fold
    fold_recall_scores[model_name] = random_search.cv_results_["mean_test_score"]

    print(f"Best parameters for {model_name}: {random_search.best_params_}")
    print(f"Best Recall score for {model_name}: {random_search.best_score_}")
    print("-" * 50)

###############################################
# Print final best parameters and Recall scores for all models
###############################################

print("Best hyperparameters and Recall scores for each model:")
for model_name, params in best_params.items():
    print(f"{model_name}: {params} with Recall = {best_recall_scores[model_name]}")


###############################################
# Plot the recall scores for each fold for each model
###############################################

fig, ax = plt.subplots(figsize=(15, 8))

for model_name, recall_scores in fold_recall_scores.items():
    ax.plot(range(1, len(recall_scores) + 1), recall_scores, label=model_name)

ax.set_title("Recall Scores for Each Fold (RandomizedSearchCV)")
ax.set_xlabel("Fold Number")
ax.set_ylabel("Recall Score")
ax.legend()
fig.tight_layout()

plt.show()

###############################################
# Plotting the Recall Scores for Each Model
###############################################

# Data for plotting
model_names = list(best_recall_scores.keys())
recall_scores = list(best_recall_scores.values())

plt.figure(figsize=(10, 6))

# Using a colormap to create a gradient of colors
colors = plt.cm.RdBu(np.linspace(0, 1, len(model_names)))

plt.barh(model_names, recall_scores, color=colors)
plt.xlabel("Recall Score")
plt.title("Recall Scores for Different Models")
plt.xlim(0, 1)  # Recall scores range from 0 to 1
plt.show()


##############################################################################################
# Best Model
###############################################

param_grid = {
    "n_estimators": np.arange(50, 500, 50),
    "max_depth": [None, 10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

# Initialize the RandomForestClassifier
rf = RandomForestClassifier()

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring="recall"
)

# Fit the grid search to the data
grid_search.fit(X_train_resampled, y_train_resampled)

# Print the best parameters and the best recall score
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best Recall: {grid_search.best_score_}")

###############################################
# Best Model With Best Hyperparameters
###############################################

# Define and train the final model with the best hyperparameters
best_model = RandomForestClassifier(
    n_estimators=150,
    min_samples_split=10,
    min_samples_leaf=1,
    max_depth=None,
    bootstrap=False,
)


# Fit the model on the training data
best_model.fit(X_train_resampled, y_train_resampled)

###############################################
# Evaluate the Best Model (After Hyperparameter Tuning)
###############################################

# Predict on the original test set using the best model
y_pred_best = best_model.predict(X_test)

# Print classification report for the best model
print("Classification Report (Best Model after Hyperparameter Tuning):")
print(classification_report(y_test, y_pred_best))

# Confusion matrix for the best model
conf_matrix_best = confusion_matrix(y_test, y_pred_best)

# Visualizing confusion matrix for the best model
plt.figure(figsize=(6, 4))
sns.heatmap(
    conf_matrix_best,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Best Model after Hyperparameter Tuning)")
plt.show()

###############################################
# ROC-AUC Curve
###############################################

# Import necessary libraries
from sklearn.metrics import roc_curve, auc

# Probabilities for ROC AUC evaluation
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()


###############################################
# Save the Final Model and Vectorizer for Future Use
###############################################

import joblib

# Save the best model and vectorizer for future predictions
joblib.dump(best_model, "./Models/best_spam_classifier_model.pkl")
joblib.dump(vectorizer, "./Models/tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")

###############################################
# Load and Test the Saved Model (Optional)
###############################################

# Load the model and vectorizer to make predictions later
loaded_model = joblib.load("./Models/best_spam_classifier_model.pkl")
loaded_vectorizer = joblib.load("./Models/tfidf_vectorizer.pkl")

# Test the saved model with a new message
new_message = ["Congratulations, you've won a $1000 gift card! Click here to claim."]
new_message_transformed = loaded_vectorizer.transform(new_message)
prediction = loaded_model.predict(new_message_transformed)

# Print prediction result (0 = Ham, 1 = Spam)
print(f"The new message is classified as: {'Spam' if prediction[0] == 1 else 'Ham'}")

###############################################
# Hurray! The END :)
###############################################
