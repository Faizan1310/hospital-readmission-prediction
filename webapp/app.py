from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'kore-clinical-outcome-risk-evaluator'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    time_in_hospital = db.Column(db.Integer)
    num_medications = db.Column(db.Integer)
    num_lab_procedures = db.Column(db.Integer)
    number_diagnoses = db.Column(db.Integer)
    insulin = db.Column(db.String(5))
    change = db.Column(db.String(5))
    risk = db.Column(db.String(20))
    probability = db.Column(db.Float)
    ai_report = db.Column(db.Text)
    ai_recommendations = db.Column(db.Text)
    date = db.Column(db.DateTime, default=datetime.utcnow)

with open('../outputs/rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

sample = pd.read_csv('../outputs/cleaned_data.csv')
feature_names = sample.drop(columns=['readmitted_30']).columns.tolist()

def generate_ai_report(patient_data, probability, risk):
    prompt = f"""You are a medical AI assistant. Based on the following patient data, generate a concise professional medical summary report.

Patient Data:
- Age Group: {patient_data['age']} (1=youngest, 9=oldest)
- Gender: {patient_data['gender']}
- Time in Hospital: {patient_data['time_in_hospital']} days
- Number of Medications: {patient_data['num_medications']}
- Number of Lab Procedures: {patient_data['num_lab_procedures']}
- Number of Diagnoses: {patient_data['number_diagnoses']}
- Insulin Given: {patient_data['insulin']}
- Medication Changed: {patient_data['change']}
- Readmission Risk: {risk} ({probability}% probability)

Write a 3-4 sentence professional medical summary explaining the patient's risk profile, contributing factors, and urgency level."""

    message = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.choices[0].message.content

def generate_recommendations(patient_data, probability, risk):
    prompt = f"""You are a medical AI assistant. Based on this patient's readmission risk profile, provide 4 specific actionable recommendations.

Patient Profile:
- Age Group: {patient_data['age']} (1=youngest, 9=oldest)
- Time in Hospital: {patient_data['time_in_hospital']} days
- Medications: {patient_data['num_medications']}
- Lab Procedures: {patient_data['num_lab_procedures']}
- Diagnoses: {patient_data['number_diagnoses']}
- Insulin: {patient_data['insulin']}
- Medication Changed: {patient_data['change']}
- Risk Level: {risk} ({probability}%)

Provide exactly 4 recommendations. Format each as:
[PRIORITY] Action: Description"""

    message = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.choices[0].message.content

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/app')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    time_in_hospital = int(request.form['time_in_hospital'])
    num_medications = int(request.form['num_medications'])
    num_lab_procedures = int(request.form['num_lab_procedures'])
    number_diagnoses = int(request.form['number_diagnoses'])
    insulin = int(request.form['insulin'])
    change = int(request.form['change'])

    full_features = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    full_features['age'] = age
    full_features['gender'] = gender
    full_features['time_in_hospital'] = time_in_hospital
    full_features['num_medications'] = num_medications
    full_features['num_lab_procedures'] = num_lab_procedures
    full_features['number_diagnoses'] = number_diagnoses
    full_features['insulin'] = insulin
    full_features['change'] = change

    probability = model.predict_proba(full_features)[0][1]
    risk = "HIGH RISK" if probability >= 0.5 else "LOW RISK"
    color = "#e53e3e" if probability >= 0.5 else "#38a169"
    prob_percent = round(probability * 100, 2)

    patient_data = {
        'age': age, 'gender': 'Male' if gender == 1 else 'Female',
        'time_in_hospital': time_in_hospital,
        'num_medications': num_medications,
        'num_lab_procedures': num_lab_procedures,
        'number_diagnoses': number_diagnoses,
        'insulin': 'Yes' if insulin == 1 else 'No',
        'change': 'Yes' if change == 1 else 'No'
    }

    ai_report = generate_ai_report(patient_data, prob_percent, risk)
    ai_recommendations = generate_recommendations(patient_data, prob_percent, risk)

    record = Prediction(
        age=age, gender=patient_data['gender'],
        time_in_hospital=time_in_hospital,
        num_medications=num_medications,
        num_lab_procedures=num_lab_procedures,
        number_diagnoses=number_diagnoses,
        insulin=patient_data['insulin'],
        change=patient_data['change'],
        risk=risk, probability=prob_percent,
        ai_report=ai_report,
        ai_recommendations=ai_recommendations
    )
    db.session.add(record)
    db.session.commit()

    importance = model.feature_importances_
    top_features = pd.Series(importance, index=feature_names).nlargest(5)
    chart_labels = top_features.index.tolist()
    chart_values = [round(v * 100, 2) for v in top_features.values.tolist()]

    return render_template('index.html',
                           prediction=risk,
                           probability=prob_percent,
                           color=color,
                           chart_labels=chart_labels,
                           chart_values=chart_values,
                           ai_report=ai_report,
                           ai_recommendations=ai_recommendations)

@app.route('/history')
def history():
    records = Prediction.query.order_by(Prediction.date.desc()).all()
    return render_template('history.html', records=records)

@app.route('/insights')
def insights():
    records = Prediction.query.all()
    if not records:
        return render_template('insights.html', insights=None)

    total = len(records)
    high_risk = sum(1 for r in records if r.risk == "HIGH RISK")
    avg_prob = round(sum(r.probability for r in records) / total, 2)

    prompt = f"""You are a medical data analyst. Analyze these hospital readmission prediction statistics and provide insights.

Statistics:
- Total Predictions: {total}
- High Risk Patients: {high_risk} ({round(high_risk/total*100, 1)}%)
- Low Risk Patients: {total - high_risk} ({round((total-high_risk)/total*100, 1)}%)
- Average Risk Probability: {avg_prob}%
- Average Medications: {round(sum(r.num_medications for r in records)/total, 1)}
- Average Hospital Stay: {round(sum(r.time_in_hospital for r in records)/total, 1)} days
- Average Diagnoses: {round(sum(r.number_diagnoses for r in records)/total, 1)}

Provide 4 key insights and trends from this data. Be specific and actionable. Format as numbered list."""

    message = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    insights_text = message.choices[0].message.content

    return render_template('insights.html',
                           insights=insights_text,
                           total=total,
                           high_risk=high_risk,
                           avg_prob=avg_prob)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    language = data.get('language', 'English')
    patient_context = data.get('patient_context', '')

    prompt = f"""You are KORE AI Assistant — a multilingual medical AI assistant for KORE (Clinical Outcome Risk Evaluator), a hospital readmission prediction system.

{f'Current Patient Context: {patient_context}' if patient_context else ''}

The user is communicating in {language}. Always respond in the same language as the user's message.

User message: {user_message}

Provide a helpful, accurate, and empathetic response. Keep it concise (2-3 sentences max)."""

    try:
        message = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"response": message.choices[0].message.content})
    except Exception as e:
        return jsonify({"response": "I'm sorry, something went wrong. Please try again."})

@app.route('/clear_history')
def clear_history():
    Prediction.query.delete()
    db.session.commit()
    return redirect(url_for('history'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
if __name__ == '__main__':
    app.run(debug=True)
    