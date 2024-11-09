import pickle
import numpy as np
from django.shortcuts import render
from django.http import HttpResponse
from tensorflow.keras.models import load_model

# Load the model and scaler
model_path = 'C:\\Users\\Abhi\\Desktop\\ModelDeploy\\myModels\\my_model.keras'
scaler_path = 'C:\\Users\\Abhi\\Desktop\\ModelDeploy\\myModels\\scaler.pkl'

# Load the model and scaler
model = load_model(model_path)
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

def home(request):
    return render(request, 'index.html')

def results(request):
    # Check if the request method is POST
    if request.method == 'POST':
        try:
            # Get data from the request
            ap_hi = float(request.POST.get('ap_hi'))
            ap_lo = float(request.POST.get('ap_lo'))
            age = float(request.POST.get('age'))
            age_years = float(request.POST.get('age_years'))
            cholesterol = float(request.POST.get('cholesterol'))
            weight = float(request.POST.get('weight'))
            bmi = float(request.POST.get('bmi'))

            # Prepare features for prediction
            features = [ap_hi, ap_lo, age, age_years, cholesterol, weight, bmi]
            features_scaled = scaler.transform([features])

            # Make prediction
            prediction = model.predict(features_scaled)
            prediction_class = np.argmax(prediction, axis=1)

            result = 'Presence' if prediction_class[0] == 1 else 'Absence'
            return render(request, 'results.html', {'result': result})

        except Exception as e:
            return HttpResponse(f"An error occurred: {e}")

    else:
        return HttpResponse("Invalid request method. Please use POST.")
