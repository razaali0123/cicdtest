import numpy as np
import pickle
from flask import Flask,request
from flask import render_template

app = Flask(__name__)

@app.route('/')
def welcome():
    return "Welcome all"


@app.route("/predict")
def taking_vars():
    a = request.args.get("a")
    b = request.args.get("b")
    c = request.args.get("c")
    d = request.args.get("d")
    filename = 'finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))
    return "The pred is " + str(model.predict(np.array([a,b,c,d]).reshape(1,-1)))
@app.route("/html")
def call():
    return render_template("index.html")

if __name__ == "__main__":
    app.run()