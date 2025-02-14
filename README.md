# SMS Spam Classification using Logistic Regression

This project demonstrates how to classify SMS messages as either "spam" or "ham" (not spam) using machine learning techniques, specifically **Logistic Regression**. The dataset used is the [SMSSpamCollection dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip), which contains SMS messages labeled as spam or ham.

### Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Model Training & Evaluation](#model-training--evaluation)
- [Model Tuning & Threshold Adjustment](#model-tuning--threshold-adjustment)
- [Final Model & Deployment](#final-model--deployment)
- [Usage](#usage)
- [Conclusion](#conclusion)

---

### Project Overview

In this project, we:
1. Download and preprocess the SMS dataset.
2. Perform exploratory data analysis (EDA) to understand the distribution and structure.
3. Train a logistic regression model to classify SMS messages.
4. Apply **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance.
5. Fine-tune the model using **GridSearchCV** to find the best hyperparameters.
6. Adjust the decision threshold to improve recall and performance.
7. Save the final model and vectorizer for future use.

---

### Installation

To run this project, you need Python installed on your system. Follow these steps to set up the environment:

1. **Clone this repository** (if applicable) or copy the code into your local environment.
2. Install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

---

### Dependencies

This project requires the following libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `requests`
- `nltk`
- `sklearn`
- `imblearn`
- `wordcloud`
- `joblib`

You can install them using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

### Data Preprocessing

The dataset is loaded and cleaned as follows:

1. **Download ZIP**: The dataset is downloaded from the UCI Machine Learning Repository.
2. **Extract and Load**: The ZIP file is extracted, and the dataset is loaded into a Pandas DataFrame.
3. **Check for Missing Data**: Missing values and duplicates are handled.
4. **Label Encoding**: The categorical labels ("ham" and "spam") are converted into numeric values (0 for ham, 1 for spam).
5. **Text Length**: A feature representing the length of each SMS message is added.
6. **Word Frequency**: Common words in spam and ham messages are extracted and analyzed.

---

### Model Training & Evaluation

The following steps were performed to train and evaluate the model:

1. **Text Vectorization**: We used **TF-IDF** (Term Frequency-Inverse Document Frequency) to convert text data into numerical features.
2. **Train-Test Split**: The dataset is split into training (80%) and testing (20%) sets.
3. **Model Training**: A **Logistic Regression** model is trained on the processed text data.
4. **Evaluation**: The model's performance is evaluated using **classification report** and **confusion matrix**.

---

### Model Tuning & Threshold Adjustment

To improve the model's performance:

1. **SMOTE**: The training data is resampled using **SMOTE** to balance the class distribution (spam vs. ham).
2. **Hyperparameter Tuning**: We performed a **GridSearchCV** to find the best regularization parameter (`C`) for the Logistic Regression model.
3. **Threshold Adjustment**: The decision threshold was adjusted from 0.5 to 0.4 to increase the recall of spam detection.

---

### Final Model & Deployment

1. **Save Model & Vectorizer**: The trained Logistic Regression model and TF-IDF vectorizer are saved using **joblib** for future predictions.
2. **Load and Predict**: The saved model and vectorizer can be reloaded to make predictions on new, unseen SMS messages.

Example:
```python
# Load the saved model
loaded_model = joblib.load("best_spam_classifier_model.pkl")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Test the saved model with a new message
new_message = ["Congratulations, you've won a $1000 gift card! Click here to claim."]
new_message_transformed = loaded_vectorizer.transform(new_message)
prediction = loaded_model.predict(new_message_transformed)

# Print prediction result
print(f"The new message is classified as: {'Spam' if prediction[0] == 1 else 'Ham'}")
```

---

### Usage

1. **Download the Dataset**: The script automatically downloads and extracts the dataset.
2. **Train Model**: The logistic regression model is trained and evaluated on the processed data.
3. **Threshold Adjustment**: The threshold for predicting spam is adjusted to optimize performance.
4. **Final Model**: The final model is saved for future use.

To run the script, execute:

```bash
python spam_classifier.py
```

---

### Conclusion

This project demonstrates a basic workflow for building a spam classifier using machine learning. The model was trained using Logistic Regression, and various techniques such as SMOTE for balancing the data and hyperparameter tuning for optimization were applied to improve performance.

The final model can be used to classify new SMS messages as spam or ham.

---

### Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip)
- [SMOTE Documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
