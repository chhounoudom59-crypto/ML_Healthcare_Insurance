
# 🏥 Medical Insurance Charges Prediction

## 📜 Overview

Welcome to the **Medical Insurance Charges Prediction** project! 🚀 This data science project leverages Machine Learning to predict medical insurance costs for individuals based on their demographic and health characteristics.

By analyzing patterns in the data, we aim to build a robust model that can accurately estimate insurance premiums, helping both insurers and customers understand cost factors.

---

## 💡 Dataset Features

The dataset contains information about various factors affecting medical charges:

| 🏷️ Feature | 📝 Description                                                            |
| ----------- | ------------------------------------------------------------------------- |
| 🎂 Age      | Age of the primary beneficiary                                            |
| ⚧️ Sex      | Gender of the insurance contractor (female, male)                         |
| ⚖️ BMI      | Body mass index (kg/m²), measuring body weight relative to height         |
| 👶 Children | Number of children / dependents covered by insurance                      |
| 🚬 Smoker   | Smoking status of the beneficiary (yes, no)                               |
| 📍 Region   | Residential area in the US (northeast, southeast, southwest, northwest)   |
| 💲 Charges  | Individual medical costs billed by health insurance (**Target Variable**) |

---

## 🛠️ Tech Stack & Libraries

This project is built using Python and the following libraries:

* 🐍 **Python**: Core programming language
* 🐼 **Pandas**: Data manipulation and analysis
* 🔢 **NumPy**: Numerical computing
* 📊 **Matplotlib & Seaborn**: Data visualization
* 🤖 **Scikit-Learn**: Data preprocessing, pipelines, and model evaluation
* 🚀 **XGBoost**: Extreme Gradient Boosting for optimized performance
* 💡 **LightGBM**: Light Gradient Boosting Machine
* 🐱 **CatBoost**: Gradient boosting on decision trees with categorical feature support

---

## ⚙️ Project Workflow

### 1. 🔍 Exploratory Data Analysis (EDA)

* **Histograms**: Distribution of age, BMI, and charges
* **Count Plots**: Categorical data analysis (smoker status, region)
* **Box Plots**: Outliers and cost comparisons (e.g., Smokers vs. Non-Smokers)
* **Correlation Heatmap**: Relationships between numerical features

### 2. 🧹 Data Preprocessing

* **Encoding**: OneHotEncoder for categorical features (sex, smoker, region)
* **Scaling**: StandardScaler for numerical features (age, BMI, children)
* **Splitting**: 80% Training / 20% Testing

### 3. 🤖 Model Selection & Tuning

We implemented and fine-tuned 5 regression algorithms using **GridSearchCV**:

* 📉 Linear Regression (Ridge)
* 🌲 Random Forest Regressor
* 🚀 XGBoost Regressor
* 💡 LightGBM Regressor
* 🐱 CatBoost Regressor

### 4. 📈 Evaluation & Visualization

* Metrics: **RMSE** and **R² Score**
* Learning Curves: Detect overfitting/underfitting
* Feature Importance: Identify factors driving costs (e.g., Smoking, BMI)
* Residual Plots: Analyze prediction errors

---

## 🚀 Key Results

* Smokers tend to have significantly higher medical charges 🚬💰
* BMI strongly correlates with charges, especially for smokers ⚖️
* **XGBoost** achieved the lowest RMSE, providing the most accurate predictions 🥇

---

## 💻 Installation & Usage

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/medical-insurance-prediction.git
```

2. **Navigate to the project directory**:

```bash
cd medical-insurance-prediction
```

3. **Install dependencies**:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost
```

4. **Run the Jupyter Notebook**:

```bash
jupyter notebook Medical.ipynb
```

---

## 🔮 Future Improvements

* 🆕 **Feature Engineering**: Interaction terms (e.g., BMI × Smoker)
* ☁️ **Deployment**: Streamlit or Flask web app
* 🧠 **Deep Learning**: Neural Networks with TensorFlow/PyTorch

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Check the **issues** page to get started.

---

## 📝 License

This project is licensed under the **MIT License**.

<div align="center"><b>⭐️ Don't forget to star this repo if you found it useful! ⭐️</b></div>  
