# Spam Detection Web App

This project is a simple web application that demonstrates how to build a spam detection system using Flask, a trained machine learning model, and a TF-IDF vectorizer. It’s designed for beginners and covers basic concepts like web forms, API endpoints, and error handling in Flask.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

The web application loads a pre-trained spam detection model along with a TF-IDF vectorizer. Users can input email text through a web form, and the app will predict whether the email is "Spam" or "Ham". Additionally, a JSON-based API endpoint is provided for integration with other applications.

<img src="Screenshot 2025-02-20 at 4.38.12 PM.png">
<img src="Screenshot 2025-02-20 at 4.38.41 PM.png">

## Features

- **Web Form Interface:** Easily input email text and view prediction results.
- **REST API Endpoint:** Submit JSON requests for spam detection.
- **Health Check Endpoint:** Quickly check if the app is running.
- **Logging:** Basic logging is implemented for debugging purposes.
- **Error Handling:** Uses Flask’s flash messaging to alert users of issues.

## Project Structure

```
.
├── app.py              # Main Flask application with endpoints.
├── templates/
│   └── index.html      # HTML template for the user interface.
├── best_spam_classifier_model.pkl   # Pre-trained spam detection model.
├── tfidf_vectorizer.pkl              # Pre-trained TF-IDF vectorizer.
└── README.md           # This file.
```

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/vigneshv1cky/Repo-for-Connie
   cd your-repo-name
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   Make sure you have the following packages installed:
   - Flask
   - joblib

   Install them using pip:

   ```bash
   pip install Flask joblib
   ```

4. **Place the Model and Vectorizer:**

   Ensure the files `best_spam_classifier_model.pkl` and `tfidf_vectorizer.pkl` are in the correct paths as specified in the code. Adjust the paths in `app.py` if needed.

## Usage

1. **Run the Application:**

   ```bash
   python app.py
   ```

2. **Access the Web Interface:**

   Open your web browser and go to `http://127.0.0.1:5000/` to view the home page. Paste your email text into the form and click the "Detect Spam" button to get a prediction.

3. **Using the API:**

   You can send a POST request to `http://127.0.0.1:5000/predict_api` with a JSON payload:

   ```json
   {
       "email_text": "Your sample email text here"
   }
   ```

4. **Health Check:**

   Visit `http://127.0.0.1:5000/health` to confirm the app is running. It will return a JSON response:

   ```json
   {
       "status": "ok"
   }
   ```

## API Endpoints

- **`/`**  
  Displays the home page with the input form.

- **`/predict`**  
  Accepts both GET and POST requests. Handles form submissions and displays the prediction result.

- **`/predict_api`**  
  A JSON-based endpoint. Accepts POST requests with an `email_text` field and returns the prediction result in JSON format.

- **`/health`**  
  A simple endpoint to check the health/status of the application.

## Troubleshooting

- **Model/Vectorizer Loading Error:**  
  Check that the file paths for `best_spam_classifier_model.pkl` and `tfidf_vectorizer.pkl` are correct. The application logs any errors during loading, so refer to the terminal output for hints.

- **No Email Text Provided:**  
  If you submit the form without any text, the app will flash an error message. Ensure you enter some text before submitting.

- **API Errors:**  
  If the API returns an error, verify that the JSON payload contains the `email_text` key.

## License

This project is provided for educational purposes and is free to use and modify. Feel free to fork, experiment, and improve it.
