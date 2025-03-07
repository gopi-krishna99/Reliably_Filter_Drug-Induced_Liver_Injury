import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load the CSV file containing the drug combinations and doses
df = pd.read_csv('large_drug_data.csv')

# Check the first few rows of the data to understand its structure
print(df.head())

# Prepare the data: Extract the features (Dose1, Dose2) and the target (DILI)
X = df[['Dose1', 'Dose2']]  # Features (drug doses)
y = df['DILI']  # Target (DILI outcome)

# Normalize the feature data (scaling the doses)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize and train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model (optional)
y_pred = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save the trained model and the scaler for future use
joblib.dump(model, 'dili_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
