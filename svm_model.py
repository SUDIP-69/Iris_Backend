from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import string

import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the trained model
with open('SVM.pickle', 'rb') as file:
    model = pickle.load(file)

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data=request.get_json()
        input_data1 = float(data['Sepal_length'])
        input_data2= float(data['Sepal_width'])
        input_data3= float(data['Petal_length'])
        input_data4= float(data['Petal_width'])
        predict_dataset=pd.DataFrame({'Sepal length': [input_data1], 'Sepal width': [input_data2], 'Petal length': [input_data3], 'Petal width':[input_data4]})
        prediction = model.predict(predict_dataset)[0]
        print(jsonify({'prediction':prediction}))
    except Exception as e:
        logging.error('Error processing request: %s', str(e))
        traceback.print_exc()
        return jsonify({'error': 'error occured'}), 500

if __name__ == '__main__':
    app.run(port=5000)
