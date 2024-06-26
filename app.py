from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('polynomial.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

app = Flask(__name__)

@ app.route('/')

def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['hours studied']
    data2 = request.form['previous scores']
    data3 = request.form['extracurricular activities']
    data4 = request.form['sleep hours']
    data5 = request.form['sample questions']
    arr = np.array([[data1, data2, data3, data4, data5]])
    test_x = scaler.transform(arr)
    pred = model.predict(test_x) 
    pred = int(pred[0])  
    return render_template('pred.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)