from flask import Flask, request, jsonify
import joblib

# Load the trained spam detection model and the TF-IDF vectorizer.
loaded_model = joblib.load("Project 7/best_spam_classifier_model.pkl")
loaded_vectorizer = joblib.load("Project 7//tfidf_vectorizer.pkl")

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    # Get JSON data from the request.
    data = request.get_json(force=True)

    # Extract the email text from the JSON payload.
    email_text = data.get("email_text")

    # Validate the input.
    if not email_text:
        return jsonify(
            {
                "error": 'No email text provided. Please include "email_text" in your JSON.'
            }
        ), 400

    try:
        # Transform the email text using the loaded TF-IDF vectorizer.
        # Make a prediction using the loaded model.
        # Interpret the prediction (0 = Ham, 1 = Spam)
        # Return the prediction as a JSON response.
        transformed_text = loaded_vectorizer.transform([email_text])
        prediction = loaded_model.predict(transformed_text)
        result = "Spam" if prediction[0] == 1 else "Ham"
        return jsonify({"prediction": result})

    except Exception as e:
        # Return an error message in case of failure.
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run the Flask app in debug mode.
    app.run(debug=True)

# curl -X POST -H "Content-Type: application/json" -d '{"email_text": "Congratulations, you'\''ve won a $1000 gift card! Click here to claim."}' http://127.0.0.1:5000/predict
