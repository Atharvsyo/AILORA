from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained ML model, vectorizer, and label encoder
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    # Render the input form page
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get symptoms input from the form
    symptoms = request.form["symptoms"]

    # Transform input using vectorizer
    symptoms_vector = vectorizer.transform([symptoms])

    # Make prediction
    prediction_encoded = model.predict(symptoms_vector)

    # Decode label to get disease name
    prediction = label_encoder.inverse_transform(prediction_encoded)[0]

    # Render the result page with the prediction
    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    # Run Flask app in debug mode
    app.run(debug=True)
