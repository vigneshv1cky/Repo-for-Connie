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

print(X)


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


###############################################
# Improve Model Performance
# Hyperparameter Tuning
###############################################

from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
param_grid = {"C": [0.01, 0.1, 1, 10, 50, 100, 250, 500, 1000]}

# Use GridSearchCV for best parameter selection
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring="f1")
grid_search.fit(X_train_resampled, y_train_resampled)

# Best model
best_model = grid_search.best_estimator_
print(f"Best Regularization Parameter: {grid_search.best_params_['C']}")

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
"""
By default, Logistic Regression predicts spam if probability > 0.5. 
We can lower this threshold to increase recall.
"""
###############################################

# Get probability predictions
y_probs = best_model.predict_proba(X_test)[:, 1]  # Probability of spam class

# Set a threshold (e.g., 0.9 instead of 0.5)
# You can play around with this threshold
new_threshold = 0.9
y_pred_adjusted = (y_probs >= new_threshold).astype(int)

# Evaluate new threshold
print("Performance with Adjusted Threshold:")
print(classification_report(y_test, y_pred_adjusted))

# Confusion matrix after adjusting the threshold
conf_matrix_best_adjusted = confusion_matrix(y_test, y_pred_adjusted)

# Visualizing confusion matrix after threshold adjustment
plt.figure(figsize=(6, 4))
sns.heatmap(
    conf_matrix_best_adjusted,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Best Model with Adjusted Threshold)")
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
