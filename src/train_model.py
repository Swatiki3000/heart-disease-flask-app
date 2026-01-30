import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from data_preprocessing import preprocess_data

# Create model folder if it does not exist
os.makedirs("model", exist_ok=True)

X_train, X_test, y_train, y_test, scaler = preprocess_data(
    "data/heart_cleaned.csv"
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, "model/heart_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
