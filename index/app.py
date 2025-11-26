from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load the trained pipeline (preprocessor + model)
MODEL_FILENAME = "rf_tuned.pkl"
with open(MODEL_FILENAME, 'rb') as file:
    pipeline = joblib.load(file)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = int(request.form['age'])
    sex = request.form['sex']
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = request.form['smoker']
    region = request.form['region']

    # Create DataFrame for pipeline
    input_df = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    # Predict using pipeline
    pred = pipeline.predict(input_df)[0]

    if pred < 0:
        result = 'Error calculating Amount!'
    else:
        result = 'Expected amount is {:.2f}'.format(pred)
    return render_template('op.html', pred=result)

if __name__ == '__main__':
    app.run(debug=True)