from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)
MODEL_PATH = Path("attrition_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "Age": int(request.form["Age"]),
        "Department": request.form["Department"],
        "JobRole": request.form["JobRole"],
        "YearsAtCompany": int(request.form["YearsAtCompany"]),
        "MonthlyIncome": int(request.form["MonthlyIncome"]),
        "JobLevel": int(request.form["JobLevel"]),
        "JobSatisfaction": int(request.form["JobSatisfaction"]),
        "WorkLifeBalance": int(request.form["WorkLifeBalance"]),
        "EnvironmentSatisfaction": int(request.form["EnvironmentSatisfaction"])
    }

    input_df = pd.DataFrame([data])
    probability = float(model.predict_proba(input_df)[0][1]) * 100
    prediction = int(model.predict(input_df)[0])

    if probability < 35:
        risk = "Low Risk"
    elif probability < 65:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return render_template(
        "result.html",
        probability=round(probability, 2),
        prediction="Likely to Leave" if prediction else "Likely to Stay",
        risk=risk,
        employee=data
    )

if __name__ == "__main__":
    app.run(debug=True)
