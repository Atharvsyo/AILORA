from flask import Flask, render_template, request
import joblib
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json

load_dotenv()

GEMINI_API_KEY = os.getenv("API_KEY")

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
        return [{
            "symptoms": symptoms,
            "response": "LLM explanation is unavailable right now. Below are the likely conditions based on your symptoms."
        }]

    diseases_text = ", ".join(diseases)

    prompt = f"""
    You are a reliable medical assistant AI.

    Input symptoms from user: "{symptoms}"
    Possible predicted diseases: {diseases_text}

    1. First, determine if the input symptoms are medically meaningful.
       If the input is random, incomplete, or non-medical (e.g. gibberish, emotional words, jokes, etc.),
       respond with this exact JSON:
       [
         {{
           "symptoms": "{symptoms}",
           "response": "The symptoms entered are not valid or medically recognizable. Please provide clear physical or health-related symptoms."
         }}
       ]

    2. If the symptoms are medically meaningful, respond ONLY with valid JSON array of objects including all possible predicted diseases in this exact format:
       [
         {{
           "Disease Name": "string",
           "Description": "short medical description",
           "Treatment": "common treatment or management steps",
           "Prevention": "preventive measures"
         }}
       ]

    Do not include any extra text or commentary outside the JSON.
    """

    model_gemini = genai.GenerativeModel("models/gemini-pro-latest")
    response = model_gemini.generate_content(prompt)
    response_text = getattr(response, "text", str(response))
    
    # Try to parse JSON response
    try:
        # Try to extract JSON from markdown code blocks if present
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        
        parsed_json = json.loads(response_text)
        return parsed_json
    except (json.JSONDecodeError, ValueError) as e:
        # If JSON parsing fails, return as fallback structure
        return [{
            "symptoms": symptoms,
            "response": response_text if response_text else "Unable to process the explanation."
        }]

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
        # Ensure explanation is a list/dict for template compatibility
        if isinstance(explanation, str):
            explanation = [{
                "symptoms": symptoms,
                "response": explanation
            }]
    except Exception as e:
        explanation = [{
            "symptoms": symptoms,
            "response": "We're unable to retrieve an explanation right now. Here are the likely conditions based on your input."
        }]
        
    print("Explanation:", explanation)
    print("Top Diseases:", top_diseases)
    print("Symptoms:", symptoms)
    
    return render_template(
        "result.html",
        prediction=top_diseases,
        explanation=explanation,
        symptoms=symptoms
    )

if __name__ == "__main__":
    app.run(debug=True)