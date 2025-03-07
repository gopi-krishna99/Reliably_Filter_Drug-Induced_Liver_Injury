from flask import Flask, request, jsonify, render_template
import pandas as pd
import io
import PyPDF2
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('dili_model.pkl')
scaler = joblib.load('scaler.pkl')

def parse_file(file):
    """Parse the uploaded file and return a DataFrame."""
    filename = file.filename
    file_extension = filename.rsplit('.', 1)[-1].lower()

    if file_extension == 'csv':
        return pd.read_csv(file)
    elif file_extension == 'txt':
        return pd.read_csv(file, delimiter='\t')
    elif file_extension == 'pdf':
        pdf_reader = PyPDF2.PdfFileReader(io.BytesIO(file.read()))
        text = ""
        for page in range(pdf_reader.numPages):
            text += pdf_reader.getPage(page).extract_text()
        data = {'Drug1': [], 'Drug2': [], 'Dose1': [], 'Dose2': [], 'DILI': []}
        lines = text.split('\n')
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    data['Drug1'].append(parts[0])
                    data['Drug2'].append(parts[1])
                    data['Dose1'].append(float(parts[2]))
                    data['Dose2'].append(float(parts[3]))
                    data['DILI'].append(int(parts[5]))
        return pd.DataFrame(data)
    else:
        raise ValueError("Unsupported file type")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        df = parse_file(file)
        X = df[['Dose1', 'Dose2']]
        X_scaled = scaler.transform(X)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        results = []
        for i, row in df.iterrows():
            drug1 = row['Drug1']
            drug2 = row['Drug2']
            probability = probabilities[i]
            result = {
                'Drug1': drug1,
                'Drug2': drug2,
                'Probability': probability,
                'Prediction': 'Do not use' if probability > 0.5 else 'Safe to use'
            }
            results.append(result)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_drugs():
    drug1 = request.form['drug1']
    drug2 = request.form['drug2']
    dose1 = float(request.form['dose1'])
    dose2 = float(request.form['dose2'])

    # Make prediction using the pre-trained model
    X = pd.DataFrame({'Dose1': [dose1], 'Dose2': [dose2]})
    X_scaled = scaler.transform(X)
    probability = model.predict_proba(X_scaled)[:, 1][0]
    prediction = 'Do not use' if probability > 0.5 else 'Safe to use'

    return jsonify([{
        'Drug1': drug1,
        'Drug2': drug2,
        'Dose1': dose1,
        'Dose2': dose2,
        'Probability': probability,
        'Prediction': prediction
    }])

if __name__ == '__main__':
    app.run(debug=True)
