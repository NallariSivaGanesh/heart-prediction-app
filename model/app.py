from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load your model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from form
    age = float(request.form['age'])
    gender = int(request.form['gender'])  # 1 or 2
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    ap_hi = float(request.form['ap_hi'])
    ap_lo = float(request.form['ap_lo'])
    cholesterol = int(request.form['cholesterol'])
    gluc = int(request.form['gluc'])
    smoke = int(request.form['smoke'])
    alco = int(request.form['alco'])
    active = int(request.form['active'])

    # Create dataframe in the SAME order as training
    input_data = pd.DataFrame([[ 
        age, gender, height, weight, ap_hi, ap_lo,
        cholesterol, gluc, smoke, alco, active
    ]], columns=[
        "age", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active"
    ])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    result = "Heart Issue Detected" if prediction == 1 else "No Heart Issue"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

