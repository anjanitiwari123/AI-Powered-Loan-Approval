# AI-Powered Loan Approval Prediction System

An intelligent web application that predicts loan approval using machine learning models trained on historical loan application data.

The system analyzes applicant information, processes it through a trained classification model, and estimates the likelihood of loan approval in real time. Multiple machine learning algorithms were evaluated to identify the best balance between predictive performance and reliable probability estimation.

---

## Features

- AI-powered loan approval prediction
- Real-time approval probability estimation
- Interactive Streamlit web application
- Automatic data preprocessing
- Multiple machine learning models compared
- High prediction accuracy
- User-friendly interface

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| Machine Learning | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |

---

## Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine
- Naive Bayes
- Gradient Boosting
- XGBoost

---

## Model Selection

Several machine learning algorithms were trained and evaluated.

| Model | Test Accuracy |
|--------|--------------:|
| XGBoost | **95.78%** |
| Logistic Regression | **90%** |

XGBoost achieved the highest classification accuracy. However, during testing its probability estimates were frequently overconfident, often producing values very close to 0% or 100%.

Since this application displays prediction confidence to end users, reliable probability estimation was an important requirement.

Logistic Regression generated more stable and interpretable approval probabilities while maintaining strong predictive performance. For this reason, Logistic Regression was selected as the deployment model.

---

## Application Workflow

1. User enters loan applicant details.
2. Input data is preprocessed.
3. The trained Logistic Regression model predicts the outcome.
4. The application displays:
   - Loan approval prediction
   - Approval probability
   - Rejection probability

---



## Future Enhancements

- Explainable AI using SHAP
- Probability calibration techniques
- REST API integration
- Cloud deployment
- Loan analytics dashboard

---

## License

This project is intended for educational and learning purposes.
