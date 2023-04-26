from flask import Flask, request, render_template
import librosa
import os
import pandas as pd
import numpy as np
import pickle

# Load the SVM model
model = pickle.load(open("svm_model.pkl", "rb"))

# Load the scaler
scaler = pickle.load(open("scaler.pkl", "rb"))


# Input the data and transform it to MFCC
def to_mfcc(f):
    arr_mfcc = 0
    y, sr = librosa.load(f)
    y_new = librosa.util.fix_length(data = y,size = 441000)

    # Compute mel-spectrogram
    mfcc = librosa.feature.mfcc(y=y_new, sr=sr, n_mfcc = 40)

    #Expand dimension
    mfcc = np.expand_dims(mfcc,axis=2)

    if type(arr_mfcc) == int:
      arr_mfcc = mfcc
    else:
      arr_mfcc = np.concatenate((arr_mfcc, mfcc), axis = 2)

    # Save it as a numpy file
    np.save('arr_mfcc',arr_mfcc)

  

# Define a function to transform the mfcc data
def transform_mfcc(file_name):
    # Load the data
    data = np.load(file_name)
    data = np.moveaxis(data, 2, 0)

    # Create a list to store the data
    data_list = []

    # Loop over each audio file
    for i in range(len(data)):
        # Loop over each time step
        time_steps = []
        for j in range(1, len(data[i])):
            time_steps.append(data[i][j])

        # Append the data for this audio file to the list
        data_list.append(time_steps)

    # Create the dataframe
    df = pd.DataFrame({'data': data_list, 'label': 0})

    # Create an X array (only the mfcc data)
    X = df['data'].values
    X = np.array([np.array(x).reshape(-1) for x in X])

    return X

# Create a Flask app
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def upload_file():
    return render_template('upload.html')

# Define a route for the prediction page
@app.route("/predict", methods=['POST'])
def predict():
    # Get the uploaded file from the request
    file = request.files['file']

    # Turn it to mfcc
    to_mfcc(file)

    # Transform the mfcc data
    arr_mfcc = transform_mfcc('arr_mfcc.npy')

    # Scale the input data
    arr_mfcc = scaler.transform(arr_mfcc)

    # Make a prediction
    prediction = model.predict(arr_mfcc)

    if prediction ==0:
        prediction = "Not Healthy"
    else:
        prediction = "Healthy"

    # Delete the arr_mfcc.npy file
    os.remove('arr_mfcc.npy')
    
    # return the prediction to the HTML template
    return render_template('upload.html', prediction=prediction)



# Run the app
if __name__ == "__main__":
    app.run()
