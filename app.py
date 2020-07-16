# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:12:27 2020

@author: PrathameshSirsikar
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['post'])
def predict():
    # ...
    # for rendering results on HTML GUI
    # ...
    
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        
        output = round(prediction[0], 2)
        
        return render_template('index.html', prediction_text= "Recommended movies should be ${}". format(output))
    
if __name__ == "__main__":
    app.run(debug=True)