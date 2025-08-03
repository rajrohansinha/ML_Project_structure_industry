# -------------------- Prediction_application.py --------------------

# Import core Flask tools
from flask import Flask, request, render_template

# Other imports for ML predictions
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize Flask application
application = Flask(__name__, template_folder="templates")
app = application   

# -------------------- ROUTES --------------------

# Home page route
@app.route('/')
def index():
    # When user visits the base URL, show index.html
    return render_template('index.html') 

# Prediction page route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # Show empty form
        return render_template('home.html')
    else:
        try:
            # Collect input from the form
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),   
                writing_score=float(request.form.get('writing_score'))
            )

            # Convert input to DataFrame
            pred_df = data.get_data_as_data_frame()
            print("\n[DEBUG] DataFrame from Form:\n", pred_df)

            # Make prediction
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            print("\n[DEBUG] Prediction Result:", results)

            # Show the result back in home.html
            return render_template('home.html', results=results[0])

        except Exception as e:
            print("[ERROR] Something went wrong:", e)
            return render_template('home.html', results="Error: Check console for details.")

# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)