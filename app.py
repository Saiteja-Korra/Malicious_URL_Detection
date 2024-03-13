import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from features import featureExtraction
import pandas as pd
import joblib
import os

d = pd.read_csv('dataset_phishing.csv')
URL = pd.DataFrame(d['url'])

#path = os.path(os.path.abspath(__file__))
#model = joblib.load(os.path.join(path,'Xgboost.pkl'))

app = Flask(__name__)
model = pickle.load(open('XgBoost_model', 'rb'))

def predict_phishing(url):
    '''
    Function to predict if the URL is phishing or legitimate
    '''
    features = []
    label = 0
    l = model.feature_names_in_
    if url not in URL['url'].values:
        features = featureExtraction(url)
    else:
        for j in l:
            features.append(d.loc[d['url'] == url, j].values[0])
    features_array = np.array(features, dtype=float).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    return "This website is not safe ⚠" if prediction == 1 else "This website is safe to go!👍🏻"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def prediction():
    '''
    Endpoint to handle URL input and return the prediction
    '''
    url = request.form['url']
    result = predict_phishing(url)
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
