# from flask import Flask, request, render_template
# import numpy as np
# import pickle

# app = Flask(__name__)

# # Model ve scaler'ı yükleme
# with open('svm_model.pkl', 'rb') as file:
#     scaler, model = pickle.load(file)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     int_features = [float(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     scaled_features = scaler.transform(final_features)
#     prediction = model.predict(scaled_features)
    
#     output = 'Diabet' if prediction[0] == 1 else 'No diabet'
#     return render_template('index.html', prediction_text='Result: {}'.format(output))

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, render_template
import numpy as np
import pickle
import matplotlib.pyplot as plt
import io
import base64
import matplotlib

# Matplotlib'in GUI kullanımını devre dışı bırakma
matplotlib.use('Agg')

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
    final_features = np.array(int_features).reshape(1, -1)
    scaled_features = scaler.transform(final_features)
    prediction = model.predict(scaled_features)
    
    output = 'Diabet' if prediction[0] == 1 else 'No diabet'

    # Grafik oluşturma
    fig, ax = plt.subplots()
    feature_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']
    ax.barh(feature_names, int_features, color='skyblue')
    ax.set_xlabel('Value')
    ax.set_title('User Values')
    
    # Grafik verisini base64 formatına dönüştürme
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode('utf-8')
    uri = 'data:image/png;base64,' + string

    plt.close(fig)  # Grafik kaynaklarını serbest bırakma

    return render_template('index.html', prediction_text='Result: {}'.format(output), plot_url=uri)

if __name__ == "__main__":
    app.run(debug=True)

