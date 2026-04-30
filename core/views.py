from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from datetime import datetime

import json
import os
import joblib
import pandas as pd

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import lime
import lime.lime_tabular

def current_datetime(request):
    now = datetime.now()
    return JsonResponse({
                "current": now
            })


# Load model once (faster)
MODEL_PATH = os.path.join(settings.BASE_DIR, 'core', 'models', 'svm_pipeline.pkl')
model = joblib.load(MODEL_PATH)


trainData = pd.read_csv(os.path.join(settings.BASE_DIR, 'core', 'models', 'train_data.csv'))

@csrf_exempt
def svmPredict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            # Convert to 2D array
            input_data = pd.DataFrame(data)#np.array([features])
            
            prediction = model.predict(input_data)
            
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            X_train_preprocessed = model.named_steps['preprocessor'].transform(trainData)
            X_test_preprocessed = model.named_steps['preprocessor'].transform(input_data)

            explainer_lime = lime.lime_tabular.LimeTabularExplainer(
                X_train_preprocessed,
                feature_names=feature_names,
                class_names=['bad', 'good'],
                mode='classification'
            )
            exp = explainer_lime.explain_instance(
                X_test_preprocessed[0], 
                model.named_steps['svm'].predict_proba
            )
            negative_features = [feat for feat, weight in exp.as_list() if weight < 0]

            #print("Features with negative impact:", negative_features)

            return JsonResponse({
                "prediction": int(prediction[0]),
                "bad_features": negative_features
            })

        except Exception as e:
            return JsonResponse({
                "error": str(e)
            }, status=500)

    return JsonResponse({"message": "Use POST request"})