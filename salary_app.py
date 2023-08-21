# Code for the Web App using Flask

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
# this creates the "Flask app"

model = pickle.load(open('salary_model.pkl', 'rb'))
# this loads the pickle saved model


# the following renders the webpage written by 'index.html'
# for some reason, this html file needs to be within the directory:
# "./template/index.html". 
# Otherwise it does not render
@app.route('/')
def home():
    return render_template('index.html')


# the following is a "post method"
# this provides some features to the model, lets users provide inputs

# because the "/predict" is here, that is connected with the "predict" button on the HTML!
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    # the request library helps with collecting the inputs provided

    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


# the main function "runs this entire flask"
if __name__ == "__main__":
    app.run(debug=True)

# after running with this python code, we get a local IP.
# copy-paste this local IP and we see that it's running! Cool
