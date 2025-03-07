# Reliably Filter Drug-Induced Liver Injury

This project aims to filter drug-induced liver injuries (DILI) by analyzing combinations of drugs based on doses and medical data. It uses machine learning (Random Forest) to predict if a drug combination is safe or harmful to the liver.

## Features
- **Predict DILI Risk**: Use pre-trained machine learning models to evaluate the risk of liver injury when combining drugs.
- **CSV & Manual Input**: Users can upload a CSV file with drug data or manually enter drug names and doses for evaluation.
- **Web Interface**: A simple web interface allows users to upload files or input data for risk assessment.
  
## Technologies Used
- **Python**: For backend logic and machine learning model.
- **Flask**: Web framework used for handling HTTP requests.
- **HTML/CSS/JavaScript**: For front-end user interface.
- **Scikit-learn**: For training and evaluating the machine learning model.
- **Pandas**: For handling and processing data.
- **PyPDF2**: For parsing PDF files.

## How to Run the Project Locally

### 1. Clone the repository:
   First, clone the repository to your local machine.
   ```bash
   git clone https://github.com/gopi-krishna99/Reliably_Filter_Drug-Induced_Liver_Injury.git
2. Install dependencies:
Install the necessary Python libraries by running:

bash
Copy
Edit
pip install -r requirements.txt
3. Run the Flask app:
Navigate to the project folder and run the app.

bash
Copy
Edit
python app.py
The web app will be accessible at http://127.0.0.1:5000/.

4. Upload a CSV or Enter Drug Names:
On the web page, you can either upload a CSV file with drug data or manually enter drug names and doses to predict the risk.

Pre-trained Model
A Random Forest model has been pre-trained on a dataset containing combinations of drugs and their DILI status. This model will be used to predict the safety of a drug combination based on the provided data.

Model Input:
The model requires two inputs:

Dose1: The dose of Drug1.
Dose2: The dose of Drug2.
Drug Names: If doses are not available, users can input drug names to make predictions based on the pre-trained model.
Model Output:
The output will show the following:

Drug1: The name of the first drug.
Drug2: The name of the second drug.
Probability: The predicted probability of liver injury for the drug combination.
Prediction: Whether the drug combination is "Safe to use" or "Do not use" based on the probability.
Data Format for CSV Upload
The CSV file should follow the structure below:

c
Copy
Edit
Drug1,Drug2,Dose1,Dose2,LiverEnzymesIncrease,DILI
Acetaminophen,Ibuprofen,500,200,Yes,1
Acetaminophen,Azithromycin,500,250,No,0
...
Model Training
The model is trained using a dataset of drug combinations with doses and liver injury status. The train_model.py script is used to train the model, and it outputs a saved model (dili_model.pkl) and a scaler (scaler.pkl) for predictions.

To train the model, run the following:

bash
Copy
Edit
python train_model.py
This will generate the necessary model files.

Contributing
If you'd like to contribute to this project, feel free to open a pull request. Please follow the contribution guidelines provided in the repository.

License
This project is open-source and available under the MIT License.

csharp
Copy
Edit

### Steps to add to your GitHub repository:

1. **Save this file as `README.md`.**
2. **Add and commit this file to your repository:**
   ```bash
   git add README.md
   git commit -m "Add README file"
   git push origin main
Let me know if you need any further adjustments!
