from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os

app = Flask(__name__)

# Load and prepare data
file_id = '1iUUEUPatYynmjqKar2YVqoGGRYDpdCcU'
url = f'https://drive.google.com/uc?id={file_id}'
heart_data = pd.read_csv(url)

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features')

        if not features or len(features) != X.shape[1]:
            return jsonify({'error': f'Expected {X.shape[1]} features.'}), 400

        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        result = {
            'prediction': int(prediction),
            'message': 'Has heart disease' if prediction else 'No heart disease'
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
