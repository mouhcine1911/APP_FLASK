from flask import Flask, render_template, redirect, url_for, request, make_response, jsonify
from sklearn.externals import joblib
import requests
import json
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():

    response = make_response(render_template("index.html"))  # ça c'est le contenu de ma page qui sera affichée
    return response

@app.route("/predict", methods=['POST']) # /predict : c'est le chemain de ma page
def predict():
    if request.method=='POST':
        #try:
        regressor = joblib.load("./linear_regression_model.pkl") # joblib permet de récuperer mes sauvegarde 
        data = dict(request.form.items())
        years_of_experience = np.array(float(data["YearsExperience"])).reshape(1, -1)
        prediction = regressor.predict(years_of_experience)
        response = make_response(render_template(
        "predicted.html",
        prediction = float(prediction)
        ))
        #except ValueError:
            #return jsonify("Please enter a number.")
        return response


if __name__ == '__main__':
    app.run(debug=True)
