from flask import Flask
from flask import json
from flask import request
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
from werkzeug import Response

app = Flask(__name__)

CORS(app)


@app.route("/")
def hello():
    return "Hello!"


@app.route("/hello_world")
def hello_world():
    return "Hello, World!"


@app.route("/training_data")
def get_training_data():
    # this is a super naive implementation just to try out Flask for a bit
    dataset_pd = pd.read_csv("data/auto-mpg.csv", sep=";", skipinitialspace=True)
    return Response(dataset_pd.to_json(orient="records"), mimetype="application/json")


@app.route("/predict")
def get_predict():
    # this is a super naive implementation just to try out Flask for a bit
    zylinder = request.args.get("zylinder", type=float)
    ps = request.args.get("ps", type=float)
    gewicht = request.args.get("gewicht", type=float)
    beschleunigung = request.args.get("beschleunigung", type=float)
    baujahr = request.args.get("baujahr", type=float)

    X_np = np.asarray([zylinder, ps, gewicht, beschleunigung, baujahr])

    regressor = pickle.load(open("target/models/Miles_per_Gallon_Regressor.pickle", "rb"))

    y_pred_np = regressor.predict(X_np.reshape(-1, 5))

    y_pred = y_pred_np.item()

    y_pred_dict = {"result": y_pred}

    res = app.response_class(
        response=json.dumps(y_pred_dict), status=200, mimetype="application/json"
    )

    return res
