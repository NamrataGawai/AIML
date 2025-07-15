import joblib
import numpy as np

# Load the trained model
model = joblib.load("model/decision_tree_model.pkl")

# Sample input (same feature order used during training)
sample = np.array([[35, 10, 75000, 2, 2.0, 1, 0, 1, 0, 1, 0, 0]])  # example values

# Predict
prediction = model.predict(sample)
print("Prediction:", "Will Buy Loan" if prediction[0] == 1 else "Will Not Buy Loan")
