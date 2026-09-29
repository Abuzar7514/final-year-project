import os
import json
import secrets

import joblib
import numpy as np
import pandas as pd

from flask import send_file
from pdf_generator import generate_medical_report, generate_image_analysis_report

from flask import Flask, render_template, request, session, redirect, url_for
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# LOAD MODELS AND DATA
# =========================================================

symptom_model = joblib.load("models/symptom_model_final.joblib")

with open("models/symptom_features.json", "r", encoding="utf-8") as file:
    symptom_features = json.load(file)

with open("models/symptom_classes.json", "r", encoding="utf-8") as file:
    symptom_classes = json.load(file)

medicine_df = pd.read_csv("models/medicine_dataset_updated.csv")
final_dataset_df = pd.read_csv("models/final_dataset_v2.csv")

medicine_df.columns = medicine_df.columns.astype(str).str.strip()
final_dataset_df.columns = final_dataset_df.columns.astype(str).str.strip()

for column in medicine_df.select_dtypes(include=["object", "string"]).columns:
    medicine_df[column] = medicine_df[column].astype(str).str.strip()

skin_model = load_model("models/skin_model_FINAL.keras")
cxr_model = load_model("models/cxr_model.keras")


# =========================================================
# DISEASE INFORMATION
# =========================================================

disease_aliases = {
    "Bronchial Asthma": "Asthma",
    "COVID": "COVID-19",
    "Flu": "Influenza",
    "TB": "Tuberculosis",
    "Tinea": "Fungal Infection",
}

disease_descriptions = {
    "Asthma": (
        "Asthma is a chronic condition that causes inflammation and narrowing "
        "of the airways, which may lead to wheezing, coughing, chest tightness, "
        "and difficulty breathing."
    ),
    "Common Cold": (
        "The common cold is usually a mild viral infection of the nose and "
        "throat that may cause a runny nose, sneezing, sore throat, and cough."
    ),
    "Dengue": (
        "Dengue is a mosquito-borne viral illness that may cause high fever, "
        "headache, body pain, rash, and, in severe cases, bleeding or shock."
    ),
    "Hepatitis A": (
        "Hepatitis A is a viral liver infection commonly spread through "
        "contaminated food or water."
    ),
    "Hepatitis B": (
        "Hepatitis B is a viral liver infection transmitted through infected "
        "blood or body fluids and may become chronic."
    ),
    "Hepatitis C": (
        "Hepatitis C is a blood-borne viral infection that can cause chronic "
        "liver inflammation and long-term liver damage."
    ),
    "Malaria": (
        "Malaria is a mosquito-borne parasitic illness that commonly causes "
        "fever, chills, sweating, headache, and weakness."
    ),
    "Pneumonia": (
        "Pneumonia is an infection of the lungs that may cause cough, fever, "
        "chest discomfort, and breathing difficulty."
    ),
    "Tuberculosis": (
        "Tuberculosis is a bacterial infection that commonly affects the lungs "
        "and requires professional testing and a complete treatment course."
    ),
    "Typhoid": (
        "Typhoid is a bacterial infection caused by Salmonella Typhi and may "
        "cause prolonged fever, weakness, and abdominal symptoms."
    ),
    "COVID-19": (
        "COVID-19 is a respiratory infection caused by SARS-CoV-2 and may range "
        "from mild symptoms to severe breathing problems."
    ),
    "Influenza": (
        "Influenza is a contagious viral respiratory illness that commonly "
        "causes fever, cough, body aches, headache, and fatigue."
    ),
    "Acne": "Acne is a common inflammatory skin condition caused by clogged pores.",
    "Psoriasis": "Psoriasis is a chronic immune-mediated condition that causes scaly skin patches.",
    "Impetigo": "Impetigo is a contagious bacterial skin infection.",
    "Fungal Infection": "A fungal skin infection is caused by fungi growing on the skin.",
}


# =========================================================
# FOLLOW-UP QUESTION CONFIGURATION
# =========================================================

FOLLOW_UP_QUESTIONS = {
    "Asthma": [
        {
            "name": "wheezing",
            "label": "Are you currently experiencing wheezing?",
            "type": "select",
            "options": ["Unknown", "No", "Yes"],
        },
        {
            "name": "night_symptoms",
            "label": "Do coughing or breathing symptoms become worse at night?",
            "type": "select",
            "options": ["Unknown", "No", "Yes"],
        },
        {
            "name": "spirometry",
            "label": "Spirometry result, if available",
            "type": "select",
            "options": ["Not tested", "Normal", "Obstructive pattern"],
        },
    ],
    "Common Cold": [
        {
            "name": "runny_nose",
            "label": "Do you have a runny or blocked nose?",
            "type": "select",
            "options": ["Unknown", "No", "Yes"],
        },
        {
            "name": "sore_throat",
            "label": "Do you have a sore throat or frequent sneezing?",
            "type": "select",
            "options": ["Unknown", "No", "Yes"],
        },
    ],
    "Dengue": [
        {
            "name": "dengue_ns1",
            "label": "Dengue NS1 or IgM result, if available",
            "type": "select",
            "options": ["Not tested", "Negative", "Positive"],
        },
        {
            "name": "platelet_count",
            "label": "Platelet count, if available (per µL)",
            "type": "number",
            "placeholder": "Example: 90000",
            "min": 0,
            "max": 1000000,
        },
        {
            "name": "bleeding_signs",
            "label": "Any unusual bleeding, red spots, or easy bruising?",
            "type": "select",
            "options": ["Unknown", "No", "Yes"],
        },
    ],
    "Malaria": [
        {
            "name": "malaria_test",
            "label": "Malaria rapid test or blood smear result",
            "type": "select",
            "options": ["Not tested", "Negative", "Positive"],
        },
        {
            "name": "cyclic_chills",
            "label": "Do fever and chills occur in repeated cycles?",
            "type": "select",
            "options": ["Unknown", "No", "Yes"],
        },
    ],
    "Typhoid": [
        {
            "name": "typhoid_test",
            "label": "Blood culture, TyphiDot, or Widal result",
            "type": "select",
            "options": ["Not tested", "Negative", "Positive"],
        },
        {
            "name": "abdominal_symptoms",
            "label": "Do you have abdominal pain, constipation, or diarrhea?",
            "type": "select",
            "options": ["Unknown", "No", "Yes"],
        },
    ],
    "Pneumonia": [
        {
            "name": "oxygen_saturation",
            "label": "Oxygen saturation (SpO₂), if available",
            "type": "number",
            "placeholder": "Example: 94",
            "min": 50,
            "max": 100,
        },
        {
            "name": "xray_infiltrate",
            "label": "Did a chest X-ray report show an infiltrate or consolidation?",
            "type": "select",
            "options": ["Not tested", "No", "Yes"],
        },
        {
            "name": "rapid_breathing",
            "label": "Are you experiencing rapid or difficult breathing?",
            "type": "select",
            "options": ["Unknown", "No", "Yes"],
        },
    ],
    "Tuberculosis": [
        {
            "name": "tb_test",
            "label": "TB test result (GeneXpert, sputum, or similar)",
            "type": "select",
            "options": ["Not tested", "Negative", "Positive"],
        },
        {
            "name": "cough_over_two_weeks",
            "label": "Has the cough lasted more than two weeks?",
            "type": "select",
            "options": ["Unknown", "No", "Yes"],
        },
        {
            "name": "night_sweats_weight_loss",
            "label": "Do you have night sweats or unexplained weight loss?",
            "type": "select",
            "options": ["Unknown", "No", "Yes"],
        },
    ],
    "Hepatitis A": [
        {
            "name": "hav_igm",
            "label": "Hepatitis A IgM result",
            "type": "select",
            "options": ["Not tested", "Negative", "Positive"],
        },
        {
            "name": "jaundice_dark_urine",
            "label": "Do you have yellow eyes/skin or dark urine?",
            "type": "select",
            "options": ["Unknown", "No", "Yes"],
        },
    ],
    "Hepatitis B": [
        {
            "name": "hbsag",
            "label": "HBsAg result",
            "type": "select",
            "options": ["Not tested", "Negative", "Positive"],
        },
        {
            "name": "jaundice_dark_urine",
            "label": "Do you have yellow eyes/skin or dark urine?",
            "type": "select",
            "options": ["Unknown", "No", "Yes"],
        },
    ],
    "Hepatitis C": [
        {
            "name": "anti_hcv",
            "label": "Anti-HCV or HCV RNA result",
            "type": "select",
            "options": ["Not tested", "Negative", "Positive"],
        },
        {
            "name": "jaundice_dark_urine",
            "label": "Do you have yellow eyes/skin or dark urine?",
            "type": "select",
            "options": ["Unknown", "No", "Yes"],
        },
    ],
}


# Each answer modifies the candidate disease score.
FOLLOW_UP_EVIDENCE = {
    "Asthma": {
        "wheezing": {"Yes": 18, "No": -8},
        "night_symptoms": {"Yes": 10, "No": -3},
        "spirometry": {"Obstructive pattern": 30, "Normal": -12},
    },
    "Common Cold": {
        "runny_nose": {"Yes": 14, "No": -7},
        "sore_throat": {"Yes": 10, "No": -4},
    },
    "Dengue": {
        "dengue_ns1": {"Positive": 45, "Negative": -25},
        "bleeding_signs": {"Yes": 16, "No": -2},
    },
    "Malaria": {
        "malaria_test": {"Positive": 45, "Negative": -25},
        "cyclic_chills": {"Yes": 16, "No": -5},
    },
    "Typhoid": {
        "typhoid_test": {"Positive": 42, "Negative": -22},
        "abdominal_symptoms": {"Yes": 12, "No": -4},
    },
    "Pneumonia": {
        "xray_infiltrate": {"Yes": 40, "No": -18},
        "rapid_breathing": {"Yes": 15, "No": -5},
    },
    "Tuberculosis": {
        "tb_test": {"Positive": 50, "Negative": -28},
        "cough_over_two_weeks": {"Yes": 18, "No": -8},
        "night_sweats_weight_loss": {"Yes": 15, "No": -5},
    },
    "Hepatitis A": {
        "hav_igm": {"Positive": 50, "Negative": -28},
        "jaundice_dark_urine": {"Yes": 13, "No": -4},
    },
    "Hepatitis B": {
        "hbsag": {"Positive": 50, "Negative": -28},
        "jaundice_dark_urine": {"Yes": 13, "No": -4},
    },
    "Hepatitis C": {
        "anti_hcv": {"Positive": 50, "Negative": -28},
        "jaundice_dark_urine": {"Yes": 13, "No": -4},
    },
}


# =========================================================
# SKIN AND X-RAY INFORMATION
# =========================================================

skin_classes = ["Acne", "Chickenpox", "Impetigo", "Psoriasis", "Tinea"]

skin_medicines = {
    "Acne": [
        ("Benzoyl Peroxide", "Apply once daily"),
        ("Clindamycin Gel", "Apply twice daily"),
    ],
    "Chickenpox": [
        ("Calamine Lotion", "Apply 2-3 times daily"),
        ("Paracetamol", "500 mg every 8 hours if advised"),
    ],
    "Impetigo": [
        ("Mupirocin Ointment", "Apply three times daily if prescribed"),
        ("Fusidic Acid Cream", "Apply twice daily if prescribed"),
    ],
    "Psoriasis": [
        ("Hydrocortisone Cream", "Use only as advised"),
        ("Salicylic Acid", "Apply as advised"),
    ],
    "Tinea": [
        ("Clotrimazole Cream", "Apply twice daily"),
        ("Terbinafine Cream", "Apply once daily"),
    ],
}

skin_info = {
    "Acne": {
        "description": "A common skin condition caused by clogged pores and inflammation.",
        "precautions": [
            "Keep the skin clean",
            "Avoid squeezing lesions",
            "Use non-comedogenic products",
        ],
    },
    "Chickenpox": {
        "description": "A viral infection that causes an itchy, blister-like rash.",
        "precautions": [
            "Avoid scratching",
            "Keep nails short",
            "Seek medical care if symptoms become severe",
        ],
    },
    "Impetigo": {
        "description": "A contagious bacterial skin infection that causes sores or blisters.",
        "precautions": [
            "Keep the affected area clean",
            "Avoid sharing towels",
            "Wash hands regularly",
        ],
    },
    "Psoriasis": {
        "description": "A chronic condition that causes red, scaly, and itchy skin patches.",
        "precautions": [
            "Avoid harsh soaps",
            "Moisturize regularly",
            "Avoid scratching affected areas",
        ],
    },
    "Tinea": {
        "description": "A fungal skin infection that commonly causes itchy, ring-shaped patches.",
        "precautions": [
            "Keep the skin dry",
            "Avoid sharing clothes or towels",
            "Use antifungal treatment as directed",
        ],
    },
}

cxr_classes = ["Normal", "Pneumonia", "Tuberculosis"]

cxr_info = {
    "Normal": {
        "description": "No major supported chest-disease pattern was detected by the model.",
        "medicines": [],
        "precautions": [
            "Maintain a healthy lifestyle",
            "Consult a clinician if symptoms continue",
            "Avoid smoking",
            "Arrange a checkup when medically indicated",
        ],
    },
    "Pneumonia": {
        "description": "A lung infection that may cause cough, fever, and breathing difficulty.",
        "medicines": [
            ("Azithromycin", "Only when prescribed by a clinician"),
            ("Paracetamol", "Use according to professional advice"),
        ],
        "precautions": [
            "Seek medical assessment",
            "Rest and maintain hydration",
            "Monitor breathing difficulty",
            "Seek urgent care for severe breathing problems",
        ],
    },
    "Tuberculosis": {
        "description": "A bacterial lung infection that requires confirmatory testing and supervised treatment.",
        "medicines": [
            ("Isoniazid + Rifampicin", "Use only under a TB treatment plan"),
            ("Pyrazinamide + Ethambutol", "Use only under specialist supervision"),
        ],
        "precautions": [
            "Consult a TB specialist",
            "Follow infection-control guidance",
            "Wear a mask when advised",
            "Complete the full prescribed treatment course",
        ],
    },
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def normalize_disease_name(disease):
    disease = str(disease).strip()

    if disease.lower() == "hepatitis a":
        disease = "Hepatitis A"

    return disease_aliases.get(disease, disease)


def build_symptom_options():
    options = [
        {
            "value": feature,
            "label": feature.replace("_", " ").title(),
        }
        for feature in symptom_features
    ]
    return sorted(options, key=lambda item: item["label"])


def get_medicine_recommendations(disease, severity):
    if medicine_df.empty or "Disease" not in medicine_df.columns:
        return []

    lookup_disease = normalize_disease_name(disease).lower()
    lookup_severity = str(severity).strip().lower()

    matches = medicine_df[
        medicine_df["Disease"].astype(str).str.strip().str.lower()
        == lookup_disease
    ]

    if matches.empty:
        return []

    if "Severity_Level" in matches.columns and lookup_severity:
        severity_matches = matches[
            matches["Severity_Level"].astype(str).str.strip().str.lower()
            == lookup_severity
        ]

        if not severity_matches.empty:
            matches = severity_matches

    recommendations = []

    for _, row in matches.iterrows():
        medicine = str(row.get("Medicine", "")).strip()
        dosage = str(row.get("Dosage", "")).strip()

        if not medicine or medicine.lower() == "nan":
            continue

        if not dosage or dosage.lower() == "nan":
            dosage = "Consult a qualified healthcare professional for dosage guidance."

        recommendations.append({
            "medicine": medicine,
            "dosage": dosage,
        })

    return recommendations



def apply_clinical_sanity_layer(selected_symptoms, disease_scores):
    """
    Apply conservative symptom-pattern adjustments to the Random Forest output.

    This does not diagnose disease or replace the ML model. It only reduces
    clearly implausible rankings and slightly strengthens diseases whose
    characteristic symptom groups are present.

    Parameters
    ----------
    selected_symptoms : list[str]
        Internal feature names selected by the user.
    disease_scores : dict[str, float]
        Disease names mapped to their original probability scores.

    Returns
    -------
    dict[str, float]
        Adjusted scores normalized so that they sum to 1.
    """
    symptoms = {
        str(symptom).strip().lower()
        for symptom in selected_symptoms
    }

    adjusted = {
        disease: max(float(score), 0.000001)
        for disease, score in disease_scores.items()
    }

    def has(*names):
        return any(name in symptoms for name in names)

    def has_all(*names):
        return all(name in symptoms for name in names)

    # -----------------------------------------------------
    # Asthma safeguard
    # -----------------------------------------------------
    # Asthma should normally have at least one respiratory feature.
    asthma_core = has(
        "symptom_breathlessness",
        "symptom_cough",
        "symptom_chest_pain",
        "symptom_mucoid_sputum",
    )

    if "Asthma" in adjusted and not asthma_core:
        adjusted["Asthma"] *= 0.18

        # Apply a stronger reduction when the symptom pattern is mainly
        # febrile, gastrointestinal, or systemic rather than respiratory.
        if has(
            "symptom_diarrhoea",
            "symptom_nausea",
            "symptom_vomiting",
            "symptom_joint_pain",
            "symptom_loss_of_smell",
            "symptom_abdominal_pain",
        ):
            adjusted["Asthma"] *= 0.45

    # -----------------------------------------------------
    # Dengue pattern
    # -----------------------------------------------------
    if "Dengue" in adjusted and has("symptom_high_fever"):
        dengue_support = sum([
            has("symptom_joint_pain"),
            has("symptom_pain_behind_the_eyes"),
            has("symptom_muscle_pain"),
            has("symptom_headache"),
            has("symptom_nausea", "symptom_vomiting"),
            has("symptom_red_spots_over_body", "symptom_skin_rash"),
        ])

        if dengue_support >= 2:
            adjusted["Dengue"] *= 1.45
        elif dengue_support == 1:
            adjusted["Dengue"] *= 1.18

    # -----------------------------------------------------
    # Malaria pattern
    # -----------------------------------------------------
    if "Malaria" in adjusted and has("symptom_high_fever"):
        malaria_support = sum([
            has("symptom_chills"),
            has("symptom_sweating"),
            has("symptom_headache"),
            has("symptom_fatigue", "symptom_lethargy", "symptom_malaise"),
            has("symptom_nausea", "symptom_vomiting", "symptom_diarrhoea"),
        ])

        if malaria_support >= 2:
            adjusted["Malaria"] *= 1.35
        elif malaria_support == 1:
            adjusted["Malaria"] *= 1.15

    # -----------------------------------------------------
    # Typhoid pattern
    # -----------------------------------------------------
    if "Typhoid" in adjusted and has("symptom_high_fever"):
        typhoid_support = sum([
            has("symptom_abdominal_pain", "symptom_belly_pain"),
            has("symptom_constipation", "symptom_diarrhoea"),
            has("symptom_headache"),
            has("symptom_loss_of_appetite"),
            has("symptom_toxic_look_(typhos)"),
            has("symptom_fatigue", "symptom_lethargy", "symptom_malaise"),
        ])

        if typhoid_support >= 2:
            adjusted["Typhoid"] *= 1.40
        elif typhoid_support == 1:
            adjusted["Typhoid"] *= 1.16

    # -----------------------------------------------------
    # Pneumonia pattern
    # -----------------------------------------------------
    if "Pneumonia" in adjusted:
        pneumonia_support = sum([
            has("symptom_cough"),
            has("symptom_breathlessness"),
            has("symptom_chest_pain"),
            has("symptom_phlegm", "symptom_rusty_sputum"),
            has("symptom_high_fever"),
        ])

        if pneumonia_support >= 3:
            adjusted["Pneumonia"] *= 1.35
        elif pneumonia_support == 2:
            adjusted["Pneumonia"] *= 1.15

    # -----------------------------------------------------
    # Tuberculosis pattern
    # -----------------------------------------------------
    if "Tuberculosis" in adjusted:
        tb_support = sum([
            has("symptom_cough"),
            has("symptom_weight_loss"),
            has("symptom_blood_in_sputum"),
            has("symptom_chest_pain"),
            has("symptom_fatigue", "symptom_malaise"),
        ])

        if has("symptom_cough") and tb_support >= 3:
            adjusted["Tuberculosis"] *= 1.35
        elif has("symptom_cough") and tb_support == 2:
            adjusted["Tuberculosis"] *= 1.15

    # -----------------------------------------------------
    # Common cold pattern
    # -----------------------------------------------------
    if "Common Cold" in adjusted:
        cold_support = sum([
            has("symptom_runny_nose"),
            has("symptom_congestion"),
            has("symptom_continuous_sneezing"),
            has("symptom_throat_irritation"),
            has("symptom_cough"),
            has("symptom_mild_fever"),
        ])

        if cold_support >= 3:
            adjusted["Common Cold"] *= 1.35
        elif cold_support == 2:
            adjusted["Common Cold"] *= 1.15

    # -----------------------------------------------------
    # Hepatitis pattern
    # -----------------------------------------------------
    hepatitis_support = sum([
        has("symptom_yellowish_skin"),
        has("symptom_yellowing_of_eyes"),
        has("symptom_dark_urine", "symptom_yellow_urine"),
        has("symptom_loss_of_appetite"),
        has("symptom_nausea", "symptom_vomiting"),
        has("symptom_abdominal_pain"),
    ])

    if hepatitis_support >= 3:
        for disease in ("Hepatitis A", "Hepatitis B", "Hepatitis C"):
            if disease in adjusted:
                adjusted[disease] *= 1.30
    elif hepatitis_support == 2:
        for disease in ("Hepatitis A", "Hepatitis B", "Hepatitis C"):
            if disease in adjusted:
                adjusted[disease] *= 1.12

    total = sum(adjusted.values())

    if total <= 0:
        return disease_scores

    return {
        disease: score / total
        for disease, score in adjusted.items()
    }


def predict_from_symptoms(selected_symptoms):
    symptom_vector = np.zeros(len(symptom_features), dtype=int)

    for symptom in selected_symptoms:
        if symptom in symptom_features:
            symptom_vector[symptom_features.index(symptom)] = 1

    original_probabilities = symptom_model.predict_proba(
        symptom_vector.reshape(1, -1)
    )[0]

    disease_scores = {}

    for index, probability in enumerate(original_probabilities):
        disease = normalize_disease_name(symptom_model.classes_[index])
        disease_scores[disease] = (
            disease_scores.get(disease, 0.0)
            + float(probability)
        )

    adjusted_scores = apply_clinical_sanity_layer(
        selected_symptoms,
        disease_scores,
    )

    ranked_predictions = sorted(
        adjusted_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:3]

    return [
        {
            "disease": disease,
            "confidence": round(score * 100, 2),
        }
        for disease, score in ranked_predictions
    ]


def follow_up_is_needed(top_predictions):
    if not top_predictions:
        return False

    top_confidence = float(top_predictions[0]["confidence"])
    second_confidence = (
        float(top_predictions[1]["confidence"])
        if len(top_predictions) > 1
        else 0
    )
    confidence_gap = top_confidence - second_confidence

    return top_confidence < 70 or confidence_gap < 15


def get_follow_up_questions(candidate_diseases):
    questions = []
    used_names = set()

    for disease in candidate_diseases:
        normalized = normalize_disease_name(disease)

        for question in FOLLOW_UP_QUESTIONS.get(normalized, []):
            if question["name"] not in used_names:
                questions.append(question)
                used_names.add(question["name"])

    return questions


def format_follow_up_answers(answers):
    """
    Convert submitted follow-up field names into the full question labels
    shown to the user, so the PDF contains proper question-and-answer rows.
    """
    question_labels = {}

    for disease_questions in FOLLOW_UP_QUESTIONS.values():
        for question in disease_questions:
            question_labels[question["name"]] = question["label"]

    formatted_answers = {}

    for field_name, answer in (answers or {}).items():
        if field_name == "csrf_token":
            continue

        answer_text = str(answer).strip()

        if not answer_text:
            continue

        label = question_labels.get(
            field_name,
            field_name.replace("_", " ").title(),
        )

        formatted_answers[label] = answer_text

    return formatted_answers


def format_symptom_for_report(symptom):
    """
    Convert internal feature names such as 'symptom_headache'
    into a clean report label such as 'Headache'.
    """
    label = str(symptom).replace("_", " ").strip()

    if label.lower().startswith("symptom "):
        label = label[len("symptom "):]

    return label.title()


def add_numeric_evidence(disease, answers):
    score = 0.0

    if disease == "Dengue":
        platelet_text = answers.get("platelet_count", "").strip()

        if platelet_text:
            try:
                platelet_count = float(platelet_text)

                if platelet_count < 50000:
                    score += 28
                elif platelet_count < 100000:
                    score += 20
                elif platelet_count < 150000:
                    score += 10
                elif platelet_count > 200000:
                    score -= 5
            except ValueError:
                pass

    if disease == "Pneumonia":
        oxygen_text = answers.get("oxygen_saturation", "").strip()

        if oxygen_text:
            try:
                oxygen = float(oxygen_text)

                if oxygen < 90:
                    score += 30
                elif oxygen < 94:
                    score += 20
                elif oxygen < 96:
                    score += 8
                elif oxygen >= 97:
                    score -= 4
            except ValueError:
                pass

    return score


def refine_predictions(top_predictions, answers):
    adjusted = []

    for item in top_predictions:
        disease = normalize_disease_name(item["disease"])
        base_score = max(float(item["confidence"]), 0.01)
        evidence_score = 0.0

        for question_name, answer_scores in FOLLOW_UP_EVIDENCE.get(
            disease,
            {},
        ).items():
            answer = answers.get(question_name, "").strip()
            evidence_score += answer_scores.get(answer, 0)

        evidence_score += add_numeric_evidence(disease, answers)

        adjusted_score = max(base_score + evidence_score, 0.01)

        adjusted.append({
            "disease": disease,
            "raw_score": adjusted_score,
        })

    total_score = sum(item["raw_score"] for item in adjusted)

    if total_score <= 0:
        return top_predictions

    refined = []

    for item in adjusted:
        confidence = round((item["raw_score"] / total_score) * 100, 2)

        refined.append({
            "disease": item["disease"],
            "confidence": confidence,
        })

    refined.sort(key=lambda item: item["confidence"], reverse=True)
    return refined


def render_symptom_result(
    selected_symptoms,
    age,
    gender,
    duration,
    severity,
    temperature,
    top_predictions,
    refinement_used=False,
    follow_up_answers=None,
):
    if not top_predictions:
        return redirect(url_for("symptoms"))

    final_prediction = normalize_disease_name(
        top_predictions[0]["disease"]
    )
    final_confidence = top_predictions[0]["confidence"]

    description = disease_descriptions.get(
        final_prediction,
        "No description is currently available.",
    )

    # Medicine safety threshold: only recommend medicines when the
    # final prediction confidence is at least 55%.
    MEDICINE_CONFIDENCE_THRESHOLD = 55.0

    medicine_warning = None

    if float(final_confidence) >= MEDICINE_CONFIDENCE_THRESHOLD:
        medicine_recommendations = get_medicine_recommendations(
            final_prediction,
            severity,
        )
    else:
        medicine_recommendations = []
        medicine_warning = (
            "The prediction confidence is not high enough to safely "
            "recommend medicines. Please consult a qualified doctor "
            "for treatment guidance."
        )

    report_symptoms = [
        format_symptom_for_report(symptom)
        for symptom in selected_symptoms
    ]

    session["medical_report"] = {
        "selected_symptoms": report_symptoms,
        "age": age,
        "gender": str(gender).title(),
        "duration": duration,
        "severity": str(severity).title(),
        "temperature": temperature,
        "final_prediction": final_prediction,
        "final_confidence": final_confidence,
        "top_predictions": top_predictions,
        "description": description,
        "medicine_recommendations": medicine_recommendations,
        "medicine_warning": medicine_warning,
        "medicine_confidence_threshold": MEDICINE_CONFIDENCE_THRESHOLD,
        "refinement_used": bool(refinement_used),
        "follow_up_answers": follow_up_answers or {},
    }

    # Mark the session as modified after updating the nested report data.
    session.modified = True

    return render_template(
        "symptom_result.html",
        selected_symptoms=selected_symptoms,
        age=age,
        gender=str(gender).title(),
        duration=duration,
        severity=str(severity).title(),
        temperature=temperature,
        top_predictions=top_predictions,
        final_prediction=final_prediction,
        final_confidence=final_confidence,
        description=description,
        medicine_recommendations=medicine_recommendations,
        medicine_warning=medicine_warning,
        medicine_confidence_threshold=MEDICINE_CONFIDENCE_THRESHOLD,
        refinement_used=refinement_used,
        follow_up_answers=follow_up_answers or {},
    )


def safe_upload(file_storage, prefix):
    if not file_storage or not file_storage.filename:
        return None

    extension = os.path.splitext(file_storage.filename)[1].lower()
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    if extension not in allowed_extensions:
        return None

    filename = f"{prefix}_{secrets.token_hex(8)}{extension}"
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file_storage.save(upload_path)
    return upload_path


# =========================================================
# HOME AND SYMPTOM ROUTES
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/symptoms")
def symptoms():
    session.pop("symptom_case", None)

    return render_template(
        "symptoms.html",
        symptom_options=build_symptom_options(),
    )


@app.route("/predict_symptoms", methods=["POST"])
def predict_symptoms():
    selected_raw = request.form.get("selected_symptoms", "")

    selected_symptoms = [
        symptom.strip()
        for symptom in selected_raw.split(",")
        if symptom.strip()
    ]

    age = request.form.get("age", "")
    gender = request.form.get("gender", "")
    duration = request.form.get("duration", "")
    severity = request.form.get("severity", "")
    temperature = request.form.get("temperature", "")

    if len(selected_symptoms) < 2:
        return render_template(
            "symptoms.html",
            symptom_options=build_symptom_options(),
            error="Please select at least two symptoms.",
        )

    top_predictions = predict_from_symptoms(selected_symptoms)

    candidate_diseases = [
        item["disease"]
        for item in top_predictions
    ]

    follow_up_questions = get_follow_up_questions(candidate_diseases)

    if follow_up_is_needed(top_predictions) and follow_up_questions:
        session["symptom_case"] = {
            "selected_symptoms": selected_symptoms,
            "age": age,
            "gender": gender,
            "duration": duration,
            "severity": severity,
            "temperature": temperature,
            "top_predictions": top_predictions,
        }

        return render_template(
            "follow_up.html",
            top_predictions=top_predictions,
            questions=follow_up_questions,
        )

    return render_symptom_result(
        selected_symptoms=selected_symptoms,
        age=age,
        gender=gender,
        duration=duration,
        severity=severity,
        temperature=temperature,
        top_predictions=top_predictions,
        refinement_used=False,
        follow_up_answers=None,
    )


@app.route("/submit_followup", methods=["POST"])
def submit_followup():
    case = session.get("symptom_case")

    if not case:
        return redirect(url_for("symptoms"))

    answers = {
        key: value
        for key, value in request.form.items()
        if key != "csrf_token"
    }

    refined_predictions = refine_predictions(
        case["top_predictions"],
        answers,
    )

    formatted_answers = format_follow_up_answers(answers)

    session.pop("symptom_case", None)

    return render_symptom_result(
        selected_symptoms=case["selected_symptoms"],
        age=case["age"],
        gender=case["gender"],
        duration=case["duration"],
        severity=case["severity"],
        temperature=case["temperature"],
        top_predictions=refined_predictions,
        refinement_used=True,
        follow_up_answers=formatted_answers,
    )


@app.route("/skip_followup", methods=["POST"])
def skip_followup():
    case = session.get("symptom_case")

    if not case:
        return redirect(url_for("symptoms"))

    session.pop("symptom_case", None)

    return render_symptom_result(
        selected_symptoms=case["selected_symptoms"],
        age=case["age"],
        gender=case["gender"],
        duration=case["duration"],
        severity=case["severity"],
        temperature=case["temperature"],
        top_predictions=case["top_predictions"],
        refinement_used=False,
        follow_up_answers=None,
    )


# =========================================================
# CHEST X-RAY ROUTES
# =========================================================

@app.route("/xray")
def xray():
    session.pop("xray_report", None)
    return render_template("xray.html")


def is_probably_xray_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    grayscale_score = (
        np.mean(np.abs(r - g))
        + np.mean(np.abs(r - b))
        + np.mean(np.abs(g - b))
    ) / 3

    contrast = np.std(arr)
    brightness = np.mean(arr)

    return (
        grayscale_score < 10
        and contrast > 45
        and 40 < brightness < 180
    )


@app.route("/predict_xray", methods=["POST"])
def predict_xray():
    upload_path = safe_upload(
        request.files.get("xray_image"),
        "xray",
    )

    if upload_path is None:
        return render_template(
            "xray_result.html",
            image_path="",
            prediction="Invalid Image",
            confidence=0,
            description=(
                "Please upload a JPG, JPEG, PNG, or WEBP chest X-ray image."
            ),
            medicines=[],
            report_available=False,
            precautions=[
                "Upload only a supported image file",
                "Use a clear frontal chest X-ray",
                "Ensure the lungs are visible",
            ],
        )

    web_path = "/" + upload_path.replace("\\", "/")

    if not is_probably_xray_image(upload_path):
        return render_template(
            "xray_result.html",
            image_path=web_path,
            prediction="Invalid Image",
            confidence=0,
            description=(
                "The uploaded image does not appear to be a valid chest X-ray."
            ),
            medicines=[],
            report_available=False,
            precautions=[
                "Upload only chest X-ray images",
                "Avoid selfies, documents, or unrelated images",
                "Use a clear frontal chest X-ray",
                "Ensure the lungs are visible",
            ],
        )

    img = image.load_img(upload_path, target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0
    predictions = cxr_model.predict(img_array, verbose=0)

    pred_class = cxr_classes[int(np.argmax(predictions))]
    confidence = float(np.max(predictions)) * 100

    if confidence < 60:
        xray_description = (
            "The uploaded image appears to be a valid chest X-ray, "
            "but the model confidence is below the required 60% threshold. "
            "The system cannot provide a reliable disease classification "
            "or medicine recommendation. Please consult a qualified "
            "healthcare professional."
        )

        xray_precautions = [
            "Do not rely on this result as a diagnosis",
            "Consult a qualified healthcare professional",
            "Use a clear frontal chest X-ray if repeating the assessment",
            "Seek prompt medical care if symptoms are severe or worsening",
        ]

        session["xray_report"] = {
            "module_name": "Chest X-ray Analysis",
            "prediction": "Uncertain X-ray Prediction",
            "confidence": round(confidence, 2),
            "description": xray_description,
            "medicines": [],
            "precautions": xray_precautions,
            "image_path": upload_path,
            "confidence_threshold": 60,
            "low_confidence": True,
        }
        session.modified = True

        return render_template(
            "xray_result.html",
            image_path=web_path,
            prediction="Uncertain X-ray Prediction",
            confidence=round(confidence, 2),
            description=xray_description,
            medicines=[],
            precautions=xray_precautions,
            report_available=True,
        )

    xray_description = cxr_info[pred_class]["description"]
    xray_medicines = cxr_info[pred_class]["medicines"]
    xray_precautions = cxr_info[pred_class]["precautions"]

    session["xray_report"] = {
        "module_name": "Chest X-ray Analysis",
        "prediction": pred_class,
        "confidence": round(confidence, 2),
        "description": xray_description,
        "medicines": xray_medicines,
        "precautions": xray_precautions,
        "image_path": upload_path,
        "confidence_threshold": 60,
        "low_confidence": False,
    }
    session.modified = True

    return render_template(
        "xray_result.html",
        image_path=web_path,
        prediction=pred_class,
        confidence=round(confidence, 2),
        description=xray_description,
        medicines=xray_medicines,
        precautions=xray_precautions,
        report_available=True,
    )


# =========================================================
# SKIN ROUTES
# =========================================================

@app.route("/skin")
def skin():
    session.pop("skin_report", None)
    return render_template("skin.html")


def is_probably_skin_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    skin_mask = (
        (r > 95)
        & (g > 40)
        & (b > 20)
        & (
            np.maximum.reduce([r, g, b])
            - np.minimum.reduce([r, g, b])
            > 15
        )
        & (np.abs(r - g) > 15)
        & (r > g)
        & (r > b)
    )

    skin_ratio = np.sum(skin_mask) / skin_mask.size
    return skin_ratio > 0.18


@app.route("/predict_skin", methods=["POST"])
def predict_skin():
    upload_path = safe_upload(
        request.files.get("skin_image"),
        "skin",
    )

    if upload_path is None:
        return render_template(
            "skin_result.html",
            image_path="",
            prediction="Invalid Image",
            confidence=0,
            medicines=[],
            report_available=False,
            description=(
                "Please upload a JPG, JPEG, PNG, or WEBP skin image."
            ),
            precautions=[
                "Upload only a supported image file",
                "Use a clear close-up image",
                "Ensure the affected area is visible",
            ],
        )

    web_path = "/" + upload_path.replace("\\", "/")

    if not is_probably_skin_image(upload_path):
        return render_template(
            "skin_result.html",
            image_path=web_path,
            prediction="Invalid Image",
            confidence=0,
            medicines=[],
            report_available=False,
            description=(
                "The uploaded image does not appear to be a clear skin image."
            ),
            precautions=[
                "Upload only skin images",
                "Avoid documents and unrelated objects",
                "Use a clear close-up image",
                "Ensure the affected area is visible",
            ],
        )

    img = image.load_img(upload_path, target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0
    predictions = skin_model.predict(img_array, verbose=0)

    pred_class = skin_classes[int(np.argmax(predictions))]
    confidence = float(np.max(predictions)) * 100

    if confidence < 60:
        skin_description = (
            f"The model predicts {pred_class}, but the confidence is below "
            "the required 60% threshold. Because the prediction confidence "
            "is low, no medicine recommendations will be provided. "
            "Please consult a qualified healthcare professional."
        )

        skin_precautions = [
            "Treat this result as uncertain",
            "Do not start medication based only on this prediction",
            "Consult a qualified healthcare professional",
            "Use a clear close-up image if repeating the assessment",
        ]

        session["skin_report"] = {
            "module_name": "Skin Disease Analysis",
            "prediction": pred_class,
            "confidence": round(confidence, 2),
            "description": skin_description,
            "medicines": [],
            "precautions": skin_precautions,
            "image_path": upload_path,
            "confidence_threshold": 60,
            "low_confidence": True,
        }
        session.modified = True

        return render_template(
            "skin_result.html",
            image_path=web_path,
            prediction=pred_class,
            confidence=round(confidence, 2),
            medicines=[],
            description=skin_description,
            precautions=skin_precautions,
            report_available=True,
            low_confidence=True,
        )

    skin_description = skin_info[pred_class]["description"]
    skin_recommendations = skin_medicines[pred_class]
    skin_precautions = skin_info[pred_class]["precautions"]

    session["skin_report"] = {
        "module_name": "Skin Disease Analysis",
        "prediction": pred_class,
        "confidence": round(confidence, 2),
        "description": skin_description,
        "medicines": skin_recommendations,
        "precautions": skin_precautions,
        "image_path": upload_path,
        "confidence_threshold": 60,
        "low_confidence": False,
    }
    session.modified = True

    return render_template(
        "skin_result.html",
        image_path=web_path,
        prediction=pred_class,
        confidence=round(confidence, 2),
        medicines=skin_recommendations,
        description=skin_description,
        precautions=skin_precautions,
        report_available=True,
        low_confidence=False,
    )


@app.route("/download_skin_report")
def download_skin_report():
    report_data = session.get("skin_report")

    if not report_data:
        return redirect(url_for("skin"))

    try:
        pdf_buffer, report_id = generate_image_analysis_report(report_data)

        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{report_id}_skin_report.pdf",
        )

    except Exception as error:
        app.logger.exception(
            "Skin report generation failed: %s",
            error,
        )

        return (
            "The skin analysis report could not be generated. "
            "Please perform the skin assessment again.",
            500,
        )


@app.route("/download_xray_report")
def download_xray_report():
    report_data = session.get("xray_report")

    if not report_data:
        return redirect(url_for("xray"))

    try:
        pdf_buffer, report_id = generate_image_analysis_report(report_data)

        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{report_id}_xray_report.pdf",
        )

    except Exception as error:
        app.logger.exception(
            "X-ray report generation failed: %s",
            error,
        )

        return (
            "The chest X-ray report could not be generated. "
            "Please perform the X-ray assessment again.",
            500,
        )


@app.route("/download_report")
def download_report():
    report_data = session.get("medical_report")

    if not report_data:
        return redirect(url_for("symptoms"))

    try:
        pdf_buffer, report_id = generate_medical_report(report_data)

        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{report_id}_medical_report.pdf",
        )

    except Exception as error:
        app.logger.exception(
            "Medical report generation failed: %s",
            error,
        )

        return (
            "The medical report could not be generated. "
            "Please perform the symptom analysis again.",
            500,
        )
# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)