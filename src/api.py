from flask import Flask
from flask_cors import CORS
import pandas as pd
from werkzeug import Response

app = Flask(__name__)

CORS(app)


@app.route("/hello")
def hello_world():
    return "Hello, World!"


@app.route("/training_data")
def get_training_data():
    # this is a super naive implementation just to try out Flask for a bit
    dataset_pd = pd.read_csv("data/auto-mpg.csv", sep=";", skipinitialspace=True)
    return Response(dataset_pd.to_json(orient="records"), mimetype="application/json")
