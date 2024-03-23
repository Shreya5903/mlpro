import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template,url_for
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

dataset = pd.read_csv('diabetes.csv')

dataset_X = dataset.iloc[:,[1, 2, 3, 4, 5, 6, 8]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)


@app.route('/')
def home():
    return render_template('about.html')


@app.route('/TEST',methods=['POST'])
def TEST():
    return render_template("predict.html")

@app.route('/predict')
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict( sc.transform(final_features) )

    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred

    return render_template('predict.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
