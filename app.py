from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os

app = Flask(__name__)
CORS(app)

# Load dataset
file_id = '1iUUEUPatYynmjqKar2YVqoGGRYDpdCcU'
url = f'https://drive.google.com/uc?id={file_id}'
heart_data = pd.read_csv(url)

# Train model
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_features = np.array(data['input']).reshape(1, -1)
        prediction = model.predict(input_features)
        result = "The person has heart disease" if prediction[0] == 1 else "The person does not have heart disease"
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
 
# Run the server


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
  