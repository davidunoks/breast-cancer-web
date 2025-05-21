from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("breast_cancer_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            features = [float(request.form[f"feature{i}"]) for i in range(1, 31)]
            prediction = model.predict([features])[0]
            result = "Positive for Breast Cancer" if prediction == 1 else "Negative for Breast Cancer"
        except Exception as e:
            result = f"Error: {e}"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
