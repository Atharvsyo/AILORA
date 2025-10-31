from flask import Flask, render_template, request
import joblib
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("API_KEY")

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GENAI_AVAILABLE = True
    except Exception:
        GENAI_AVAILABLE = False

app = Flask(__name__)

def load_artifacts():
    try:
        loaded_model = joblib.load("latest_model.pkl")
        loaded_vectorizer = joblib.load("latest_vectorizer.pkl")
        loaded_label_encoder = joblib.load("new_latest_encoder.pkl")
        return loaded_model, loaded_vectorizer, loaded_label_encoder
    except Exception as exc:
        raise RuntimeError(f"Failed to load model artifacts: {exc}")


# Load your trained ML model files
model, vectorizer, label_encoder = load_artifacts()

def get_gemini_explanation(diseases, symptoms):
    if not GENAI_AVAILABLE:
        return (
            "LLM explanation is unavailable right now. Below are the likely conditions based on "
            "your symptoms."
        )

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
    return getattr(response, "text", str(response))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    symptoms = request.form.get("symptoms", "").strip()
    if not symptoms or len(symptoms) < 3:
        return render_template("index.html", error="Please enter meaningful symptoms.")

    input_data = vectorizer.transform([symptoms])
    proba = model.predict_proba(input_data)[0]

    top_indices = proba.argsort()[-5:][::-1]
    top_diseases = label_encoder.inverse_transform(top_indices)

    try:
        explanation = get_gemini_explanation(top_diseases, symptoms)
    except Exception:
        explanation = (
            "We're unable to retrieve an explanation right now. Here are the likely "
            "conditions based on your input."
        )

    return render_template(
        "result.html",
        prediction=top_diseases,
        explanation=explanation
    )


if __name__ == "__main__":
    app.run(debug=True)
