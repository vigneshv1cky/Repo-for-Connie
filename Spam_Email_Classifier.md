# Spam Email Classifier Project

This guide walks you through the process of building a spam email classifier using machine learning. We will download a dataset, preprocess the text, train a machine learning model, evaluate its performance, and save the final model for future use.

## 1. **Download & Extract the Dataset**

First, we need to download and extract the dataset containing text messages labeled as "spam" or "ham" (non-spam). This dataset is publicly available.

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

# Load the dataset (it's inside the extracted folder)
df = pd.read_csv("spam_dataset/SMSSpamCollection", sep="\t", header=None, names=["label", "text"])

# Display first few rows
print(df.head())
```

## 2. **Check Dataset Overview**

We can check the size, structure, and types of data in the dataset.

```python
# Check dataset size and structure
print(df.shape)  # Rows & Columns
print(df.info())  # Data types & Missing values
```

## 3. **Check Class Distribution (Spam vs. Ham)**

We need to visualize the distribution of "spam" and "ham" (non-spam) messages.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Count plot of spam vs ham messages
sns.countplot(x=df["label"], palette="coolwarm")
plt.title("Spam vs. Ham Distribution")
plt.show()

# Print counts
print(df["label"].value_counts())
```

## 4. **Check for Duplicate Messages**

Sometimes, datasets contain duplicate messages. We check and remove duplicates if necessary.

```python
print("Duplicate messages:", df.duplicated().sum())
df = df.drop_duplicates()  # Remove duplicates
print("After removing duplicates:", df.shape)
```

## 5. **Check for Missing Values**

We check if there are any missing values in the dataset.

```python
print("Missing values:\n", df.isnull().sum())
```

## 6. **Convert Labels to Numeric**

Machine learning models require numerical labels, so we convert the "ham" and "spam" labels to `0` and `1`, respectively.

```python
df["label"] = df["label"].map({"ham": 0, "spam": 1})
```

## 7. **Check Text Length Distribution**

To understand the length of the messages, we plot the distribution of text lengths.

```python
df["text_length"] = df["text"].apply(len)

# Histogram of text length distribution
plt.figure(figsize=(10, 5))
sns.histplot(df[df["label"] == 0]["text_length"], bins=30, kde=True, label="Ham", color="blue")
sns.histplot(df[df["label"] == 1]["text_length"], bins=30, kde=True, label="Spam", color="red")
plt.legend()
plt.title("Text Length Distribution (Spam vs. Ham)")
plt.show()
```

## 8. **Most Common Words in Spam vs. Ham**

We extract the most common words from spam and ham messages, excluding common stopwords.

```python
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Function to get common words
def get_most_common_words(texts, n=20):
    words = " ".join(texts).lower().translate(str.maketrans("", "", string.punctuation)).split()
    words = [word for word in words if word not in stop_words]
    return Counter(words).most_common(n)

# Get common words for ham and spam
ham_words = get_most_common_words(df[df["label"] == 0]["text"])
spam_words = get_most_common_words(df[df["label"] == 1]["text"])

print("Most common words in HAM messages:", ham_words)
print("Most common words in SPAM messages:", spam_words)
```

## 9. **Word Cloud for Spam and Ham Messages**

Word clouds help visualize the most frequent words in spam and ham messages.

```python
from wordcloud import WordCloud

# Generate word clouds
spam_text = " ".join(df[df["label"] == 1]["text"])
ham_text = " ".join(df[df["label"] == 0]["text"])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(WordCloud(width=500, height=300, background_color="white").generate(spam_text))
plt.axis("off")
plt.title("Spam Word Cloud")

plt.subplot(1, 2, 2)
plt.imshow(WordCloud(width=500, height=300, background_color="white").generate(ham_text))
plt.axis("off")
plt.title("Ham Word Cloud")

plt.show()
```

## 10. **Text Preprocessing and Vectorization**

Next, we convert the text data into a numerical format using **TF-IDF Vectorization**, which represents words by their frequency across documents.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text data to numerical representation using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)  # Use top 5000 words
X = vectorizer.fit_transform(df["text"])  # Transform text into TF-IDF features
y = df["label"]  # Target variable (spam=1, ham=0)

print(X)
```
## TF-IDF Vectorization

### What is `TfidfVectorizer`?

`TfidfVectorizer` (Term Frequency-Inverse Document Frequency Vectorizer) is a method for converting text data into numerical form, which is essential for machine learning models. It assigns weights to words based on their importance in the dataset.

### Components of TF-IDF

1. **Term Frequency (TF):** Measures how often a word appears in a document.
   $$
   TF = \frac{\text{Number of times a term appears in a document}}{\text{Total number of terms in the document}}
   $$

2. **Inverse Document Frequency (IDF):** Reduces the weight of commonly occurring words and increases the weight of rare words.
   $$
   IDF = \log\left(\frac{\text{Total number of documents}}{\text{Number of documents containing the term}}\right)
   $$

### TF-IDF Score Calculation

The final **TF-IDF score** for a term is computed as:
   $$
   TF\text{-}IDF = TF \times IDF
   $$

This scoring method helps in focusing on important words while ignoring commonly occurring ones like "the", "is", etc.

### Why use `TfidfVectorizer`?

- **Handles varying document lengths:** Unlike simple word counts, TF-IDF normalizes word frequency.
- **Reduces importance of common words:** Unlike `CountVectorizer`, which treats every word equally, TF-IDF reduces the weight of frequently occurring words.
- **Works well with sparse data:** Most NLP models require numerical inputs, and TF-IDF provides a meaningful representation.
- **Better performance in spam classification:** Helps identify unique words in spam messages while ignoring commonly used words.

---


## 11. **Split Data for Training & Testing**

We split the dataset into training and testing sets (80% train, 20% test).

```python
from sklearn.model_selection import train_test_split

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")
```

## 12. **Class Distribution in Training & Testing Sets**

Check the class distribution (spam vs ham) in both the training and testing sets.

```python
# Check distribution in training and testing sets
train_distribution = y_train.value_counts(normalize=True) * 100  # Percentage format
test_distribution = y_test.value_counts(normalize=True) * 100  # Percentage format

print("Class distribution in Training Set:")
print(train_distribution)

print("\nClass distribution in Testing Set:")
print(test_distribution)
```

## 13. **Apply SMOTE for Oversampling**

## **What is SMOTE (Synthetic Minority Over-sampling Technique)?**  

SMOTE is a technique used to address the problem of class imbalance in classification tasks. In datasets where one class (e.g., spam emails) is significantly underrepresented compared to another (e.g., ham emails), traditional machine learning models tend to be biased toward the majority class. SMOTE helps by artificially generating synthetic examples of the minority class rather than simply duplicating existing ones.

---

## **Why Use SMOTE?**  

- **Handles Class Imbalance**: Machine learning models tend to perform poorly when the dataset is imbalanced because they are biased toward the majority class.
- **Avoids Overfitting**: Unlike naive oversampling (where minority class examples are duplicated), SMOTE creates new synthetic data points, reducing the risk of overfitting.
- **Enhances Generalization**: By providing additional data points for the minority class, SMOTE helps models learn more meaningful decision boundaries.

---

## **How Does SMOTE Work?**  

SMOTE generates synthetic examples by interpolating between existing samples of the minority class. The process follows these steps:

1. **Identify k-nearest neighbors**  
   - For each sample in the minority class, SMOTE finds its k-nearest neighbors (typically \( k = 5 \)) in the feature space.

2. **Select a random neighbor**  
   - One of the k-nearest neighbors is randomly selected.
   
3. **Create a synthetic data point**  
   - A new sample is created by interpolating between the original data point and the selected neighbor using the formula:  

     ```
     x_new = x_original + λ * (x_neighbor - x_original)
     ```

     where `λ` (lambda) is a random number between 0 and 1.

4. **Repeat for multiple samples**  
   - This process is repeated until the desired balance between classes is achieved.


```python
from imblearn.over_sampling import SMOTE

# Initialize SMOTE
smote = SMOTE(sampling_strategy="auto", random_state=42)

# Apply SMOTE only on training data (to avoid data leakage)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check new class distribution
print("Class distribution after SMOTE:")
print(y_train_resampled.value_counts(normalize=True) * 100)
```

### **Key Points in the Code:**
- **`sampling_strategy="auto"`**: Balances the minority class to match the majority class.
- **`random_state=42`**: Ensures reproducibility of results.
- **`fit_resample(X_train, y_train)`**: Generates new synthetic samples for the training data.
- **Avoids data leakage**: SMOTE is only applied to the training set, ensuring that synthetic data does not contaminate the test set.

---

## **When to Use SMOTE?**
✅ When the dataset is highly imbalanced.  
✅ When using machine learning models that are sensitive to class distribution (e.g., Logistic Regression, Decision Trees).  
✅ When you want to improve the recall for the minority class while maintaining model performance.  

🚫 Avoid using SMOTE when:  
❌ The dataset is already balanced.  
❌ The dataset is too small, and synthetic data might introduce noise.  
❌ Using deep learning models, as they often have built-in techniques for handling imbalance (e.g., class weighting).  


## 14. **Train Logistic Regression Model**

We now train a logistic regression model on the resampled (balanced) training data.

```python
from sklearn.linear_model import LogisticRegression

# Train logistic regression model on the balanced dataset
model_smote = LogisticRegression()
model_smote.fit(X_train_resampled, y_train_resampled)

# Predict on the original test set
y_pred_smote = model_smote.predict(X_test)
```

## 15. **Evaluate the Model**

We evaluate the performance of the model using metrics like precision, recall, F1-score, and the confusion matrix.

```python
from sklearn.metrics import classification_report, confusion_matrix

# Print evaluation metrics
print("Classification Report after SMOTE:")
print(classification_report(y_test, y_pred_smote))

# Confusion matrix
conf_matrix_smote = confusion_matrix(y_test, y_pred_smote)

# Visualizing confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_smote, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix after SMOTE")
plt.show()
```

## 16. **Hyperparameter Tuning**

We can improve the model by tuning hyperparameters. Here, we use Grid Search to find the best regularization parameter (`C`) for logistic regression.

```python
from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
param_grid = {"C": [0.01, 0.1, 1, 10, 100]}

# Use GridSearchCV for best parameter selection
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring="f1")
grid_search.fit(X_train_resampled, y_train_resampled)

# Best model
best_model = grid_search.best_estimator_
print(f"Best Regularization Parameter: {grid_search.best_params_['C']}")
```

## 17. **Save the Final Model**

Once we have the best model, we save it along with the vectorizer for future use.

```python
import joblib

# Save the best model and vectorizer for future predictions
joblib.dump(best_model, "best_spam_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")
```

## 18. **Load and Test the Saved Model**

Finally, we can load the saved model and vectorizer to make predictions on new messages.

```python
# Load the model and vectorizer to make predictions later
loaded_model = joblib.load("best_spam_classifier_model.pkl")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Test the saved model with a new message
new_message = ["Congratulations, you've won a $1000 gift card! Click here to claim."]
new_message_transformed = loaded_vectorizer.transform(new_message)
prediction = loaded_model.predict(new_message_transformed)

# Print prediction result (0 = Ham, 1 = Spam)
print(f"The new message is classified as: {'Spam' if prediction[0] == 1 else 'Ham'}")
```

## **Conclusion**

This entire process walks through downloading a dataset, processing the text data, balancing the classes, and training a logistic regression model for spam classification.

You can explore this approach by modifying different aspects like text preprocessing, model choice, or evaluation metrics to improve the performance.

### Further Reading:
- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)



