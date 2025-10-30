from flask import Flask, render_template, request
import joblib
import google.generativeai as genai

# ✅ Configure Gemini API
genai.configure(api_key="")

app = Flask(__name__)

# ✅ Load your trained ML model files
model = joblib.load("latest_model.pkl")
vectorizer = joblib.load("latest_vectorizer.pkl")
label_encoder = joblib.load("new_latest_encoder.pkl")


def get_gemini_explanation(diseases, symptoms):
    diseases_text = ", ".join(diseases)

    prompt = f"""
    User Symptoms: {symptoms}
    Predicted Diseases: {diseases_text}

    For each disease provide strictly in this format:
    Disease Name:
    Description:
    Treatment:
    Prevention:
    ---

    Keep it medically correct and short.
    Do not add emojis.
    """

    model_gemini = genai.GenerativeModel("models/gemini-pro-latest")
    response = model_gemini.generate_content(prompt)
    return response.text


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    symptoms = request.form["symptoms"]

    input_data = vectorizer.transform([symptoms])
    proba = model.predict_proba(input_data)[0]

    top_indices = proba.argsort()[-5:][::-1]
    top_diseases = label_encoder.inverse_transform(top_indices)

    explanation = get_gemini_explanation(top_diseases, symptoms)

    return render_template(
        "result.html",
        prediction=top_diseases,
        explanation=explanation
    )


if __name__ == "__main__":
    app.run(debug=True)
