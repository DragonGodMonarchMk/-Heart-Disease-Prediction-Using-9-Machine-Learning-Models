# -Heart-Disease-Prediction-Using-9-Machine-Learning-Models
A comprehensive machine learning project using 9 classification algorithms to predict heart disease based on patient health metrics. Includes performance evaluation, visualizations, and model comparison.
# 🫀 Heart Disease Prediction Using 9 Machine Learning Models

This project implements and compares **9 different classification algorithms** to predict the presence of heart disease in patients using clinical and physiological data. It serves as a comprehensive benchmarking task for classification models and health data analysis.

## 🚀 Features

- Data preprocessing and exploratory data analysis (EDA)
- Implementation of 9 ML models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Naive Bayes
  - XGBoost
  - Gradient Boosting
  - AdaBoost
- Model evaluation using:
  - Accuracy
  - Confusion Matrix
  - ROC-AUC Curve
  - Classification Report
- Comparison of model performance
- Visualizations for better insights

## 🧠 Technologies Used

- Python
- Jupyter Notebook
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- XGBoost

## 📊 Dataset

- **Source**: UCI Heart Disease dataset
- **Attributes**:
  - Age, Sex, Chest pain type, Resting BP, Cholesterol, Fasting blood sugar, ECG results, Max HR, Exercise-induced angina, Oldpeak, Slope, CA, Thal, and Target (presence of heart disease)

## 📈 Results

Each model was trained and evaluated on the dataset, and their performances were compared visually and statistically. Top performers were ensemble models such as **XGBoost** and **Random Forest**.

## 🖼 Sample Output Visuals

- ROC-AUC Curves
- Confusion Matrices
- Model comparison bar charts

## 🏁 How to Run

1. Clone the repository:
   ```bash


pip install -r requirements.txt
jupyter notebook heart-disease-prediction-using-9-models.ipynb
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction
