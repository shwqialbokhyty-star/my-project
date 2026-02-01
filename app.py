from flask import Flask, render_template, request
import joblib
import numpy as np


app = Flask(__name__)

model = joblib.load("diabetes_model.pkl")



@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        data = [
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            float(request.form["SkinThickness"]),
            float(request.form["BMI"]),
            float(request.form["DPF"]),   
            float(request.form["Age"])
        ]

        prediction = model.predict([data])[0]

        if prediction == 1:
            result = "⚠️ يوجد احتمال الإصابة بمرض السكري"
        else:
            result = "✅ لا يوجد احتمال إصابة بمرض السكري"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
