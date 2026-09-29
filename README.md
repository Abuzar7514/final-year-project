# AI Healthcare Assistant

A multimodal healthcare-support web application developed as my Final Year Project for the Bachelor of Science in Computer Engineering. The system combines symptom-based machine learning, skin-image classification, and chest X-ray classification in a single Flask application.

## Features

### Symptom-Based Disease Prediction

- Uses a Random Forest classifier to analyse symptoms.
- Predicts among 10 supported diseases.
- Asks follow-up questions when additional information is required.
- Displays prediction confidence, precautions, and medicine information.
- Generates a downloadable PDF medical report.

Supported diseases:

- Bronchial Asthma
- Common Cold
- Dengue
- Hepatitis A
- Hepatitis B
- Hepatitis C
- Malaria
- Pneumonia
- Tuberculosis
- Typhoid

### Skin Disease Classification

- Uses a MobileNetV2-based deep-learning model.
- Accepts an uploaded skin image.
- Classifies five skin conditions:
  - Acne
  - Chickenpox
  - Impetigo
  - Psoriasis
  - Tinea

### Chest X-Ray Classification

- Uses an EfficientNetB0-based deep-learning model.
- Accepts an uploaded chest X-ray.
- Classifies the image as:
  - Normal
  - Pneumonia
  - Tuberculosis

### Report Generation

- Generates structured PDF reports for symptom predictions and image analyses.
- Includes the result, confidence information, precautions, and relevant guidance.

## Technology Stack

- Python
- Flask
- TensorFlow / Keras
- scikit-learn
- NumPy
- pandas
- Pillow
- ReportLab
- HTML
- CSS
- JavaScript

## Project Structure

```text
health_ai_web_app/
├── app.py
├── pdf_generator.py
├── models/
│   ├── cxr_model.keras
│   ├── skin_model_FINAL.keras
│   └── symptom_model_final.joblib
├── static/
├── templates/
├── requirements.txt
├── .gitattributes
├── .gitignore
└── README.md
```

## Running the Project Locally

### Requirements

- Python 3.12
- Git LFS

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Abdur-Rehman-170/ai-healthcare-assistant.git
cd ai-healthcare-assistant
```

2. Download the Git LFS model files:

```bash
git lfs install
git lfs pull
```

3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate it on Windows:

```powershell
.venv\Scripts\activate
```

5. Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Start the Flask application:

```bash
python app.py
```

7. Open the application in a browser:

```text
http://127.0.0.1:5000
```

## Important Notice

This application was created for academic and educational purposes. Its predictions and generated reports are not substitutes for diagnosis, medical advice, or treatment from a qualified healthcare professional.

## Author

**Abdur Rehman**

- GitHub: [Abdur-Rehman-170](https://github.com/Abdur-Rehman-170)
- LinkedIn: [Abdur Rehman](https://www.linkedin.com/in/abdur-rehman-5760783a9)
- Email: [workbyabdurrehman@gmail.com](mailto:workbyabdurrehman@gmail.com)