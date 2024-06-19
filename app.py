from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Model ve scaler'ı yükleme
with open('svm_model.pkl', 'rb') as file:
    scaler, model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    scaled_features = scaler.transform(final_features)
    prediction = model.predict(scaled_features)
    
    output = 'Diabet' if prediction[0] == 1 else 'No diabet'
    return render_template('index.html', prediction_text='Result: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
