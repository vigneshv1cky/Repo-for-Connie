from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained spam detection model and the TF-IDF vectorizer.
try:
    loaded_model = joblib.load(
        "/Users/vignesh/GitHub/Repo-for-Connie/Project 7/flask_api/best_spam_classifier_model.pkl"
    )
    loaded_vectorizer = joblib.load(
        "/Users/vignesh/GitHub/Repo-for-Connie/Project 7/flask_api/tfidf_vectorizer.pkl"
    )
    logging.info("Model and vectorizer loaded successfully.")
except Exception as e:
    logging.error("Error loading model or vectorizer: %s", e)
    raise

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for flash messages


@app.route("/")
def index():
    # Render a home page with a form to input email text.
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        email_text = request.form.get("email_text")
        if not email_text:
            flash("No email text provided. Please input your email text.")
            return redirect(url_for("index"))
        try:
            # Transform the email text using the loaded TF-IDF vectorizer.
            # Make a prediction using the loaded model.
            # Interpret the prediction (0 = Ham, 1 = Spam)
            transformed_text = loaded_vectorizer.transform([email_text])
            prediction = loaded_model.predict(transformed_text)
            result = "Spam" if prediction[0] == 1 else "Ham"
            return render_template(
                "index.html", prediction=result, email_text=email_text
            )
        except Exception as e:
            logging.error("Prediction error: %s", e)
            flash("An error occurred during prediction. Please try again.")
            return redirect(url_for("index"))
    else:
        # For GET request, render the form.
        return render_template("index.html")


@app.route("/predict_api", methods=["POST"])
def predict_api():
    # JSON based prediction endpoint.
    data = request.get_json(force=True)
    email_text = data.get("email_text")
    if not email_text:
        return jsonify(
            {
                "error": 'No email text provided. Please include "email_text" in your JSON.'
            }
        ), 400

    try:
        transformed_text = loaded_vectorizer.transform([email_text])
        prediction = loaded_model.predict(transformed_text)
        result = "Spam" if prediction[0] == 1 else "Ham"
        return jsonify({"prediction": result})
    except Exception as e:
        logging.error("API prediction error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    # A simple health check endpoint.
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
