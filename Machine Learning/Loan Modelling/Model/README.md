# 🏦 Loan Modelling with Decision Tree

This project predicts whether a customer will buy a personal loan based on demographic and financial features.

## 🔍 Model
- Algorithm: Decision Tree Classifier
- Input Features: Age, Income, Education, Account status, and City (one-hot)
- Target: 1 = Will buy personal loan, 0 = Will not

## 🧪 Try the Model
### 🖥️ Option 1: Run CLI Script
```bash
pip install -r requirements.txt
python predict.py
```

### 🌐 Option 2: Launch Web App
```bash
python app.py
```

## 📁 File Overview
- `model/decision_tree_model.pkl`: Trained model (you add this)
- `predict.py`: Run predictions via script
- `app.py`: Interactive Gradio UI
- `requirements.txt`: Install dependencies
