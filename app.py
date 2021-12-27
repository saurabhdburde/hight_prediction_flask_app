from flask import Flask , render_template, request
import numpy as np
import pandas as pd
from joblib import load
import plotly.express as px
import plotly.graph_objects as go
import uuid


app = Flask(__name__)



def make_picture(training_data_filename, regressor, test_np, output_file):
    """ Make Plot of Predicted Age with Plotly"""
    # Impoting the dataset
    dataset = pd.read_pickle(training_data_filename)

    # Only consider dat with age more than 0
    dataset = dataset[dataset['Age'] > 0]

    ages = dataset['Age'] # Feature Series
    heights = dataset['Height'] # DV Series

    # Create new feature age X and predict it
    X_new = np.array(list(range(19))).reshape(19,1)
    preds = regressor.predict(X_new)

    # Visualising the Training set results
    fig = px.scatter(x=ages, y=heights,title = 'Ages And Heights' ,labels ={'x' : 'Age (years)', 'y' : 'Height (inches)'})
    fig.add_trace(go.Scatter(x = X_new.reshape(19), y = preds, mode='lines', name='Model'))

    # New Data
    new_preds = regressor.predict(test_np)

    # Plot data with Plotly
    fig.add_trace(go.Scatter(x = test_np.reshape(len(test_np)), y = new_preds, name='New Outputs', mode='markers', marker=dict(color='purple', size=12, line=dict(color='purple', width=2))))
    
    # Write Image
    fig.write_image(output_file, width=800, engine='kaleido')
    fig.show()

    print(new_preds)


def floats_string_to_np_arr(floats_str):
    """Conver List of Floats into np Array"""
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False

    floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
    return floats.reshape(len(floats), 1)



@app.route('/', methods=['GET','POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', href='static/base_pic.svg')
    else:

        # Got Text from Forms of HTML (request.form)
        text = request.form["text"]

        random_string = uuid.uuid4().hex

        # Path for
        path = "static/"+ random_string +".svg"

        # Loding model file
        model = load('model/Lin_reg.joblib')

        # Converting floats to np_array
        np_arr = floats_string_to_np_arr(text)

        # Making Picture Using make Picture Function (By providing all the inputs)
        make_picture('dataset/AgesAndHeights.pkl', model, np_arr,path)
        
        return render_template('index.html', href=path)



