import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
application = Flask(__name__)
app = application
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))
@app.route("/")

def index():
    return render_template('index.html')

@app.route("/predictdata", methods = ['GET','POST'])
def predict_data():
    if request.method == "POST":
        f1 = float(request.form.get('f1'))
        f2 = float(request.form.get('f2'))
        f3 = float(request.form.get('f3'))
        f4 = float(request.form.get('f4'))
        f5 = float(request.form.get('f5'))
        f6 = float(request.form.get('f6'))
        f7 = float(request.form.get('f7'))
        f8 = float(request.form.get('f8'))
        f9 = float(request.form.get('f9'))

        input_data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9]])

        new_data_scaled = standard_scaler.transform(input_data)
        result = ridge_model.predict(new_data_scaled)
        return render_template('home.html', results = result[0], )
    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host = "0.0.0.0")