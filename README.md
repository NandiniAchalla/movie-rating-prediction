# 🎬 Movie Rating Prediction using PySpark

## 📘 Overview
This project predicts movie ratings based on various features such as user preferences, genre, and the number of existing ratings.  
It demonstrates how **PySpark** can be leveraged for large-scale data processing and **machine learning**, combined with **Flask** and **Streamlit** for an interactive user interface.

The project integrates **data engineering**, **ML modeling**, and **frontend visualization**, showcasing end-to-end data pipeline development.

## ⚙️ Tech Stack
- **Programming Language:** Python  
- **Frameworks:** PySpark, Flask, Streamlit  
- **Libraries:** pandas, NumPy, scikit-learn, matplotlib  
- **Tools:** Google Colab, VS Code  
- **Database (if applicable):** CSV / Local Storage  

## 🧠 Objective
- Build a predictive model to estimate movie ratings using PySpark MLlib.  
- Handle large datasets efficiently with Spark’s distributed computing.  
- Develop a user-friendly interface for live predictions and data visualization.

## 🧩 Implementation Steps

### 1. **Data Preprocessing**
- Imported and cleaned movie metadata and user ratings datasets.  
- Merged datasets and encoded categorical features (genres, directors, etc.).  
- Split the data into training and testing sets using PySpark’s DataFrame API.

### 2. **Feature Engineering**
- Created new features such as:
  - `rating_count` (number of ratings per movie)  
  - `avg_user_rating` (average rating by the user)  
  - `genre_encoded` (numerical representation of genre)  

### 3. **Model Building**
- Used **Random Forest Regressor** from PySpark MLlib to predict numeric ratings.  
- Also tested classification mode (e.g., predicting “good” vs “bad” movie).  
- Tuned hyperparameters using Spark’s `ParamGridBuilder` and `CrossValidator`.

### 4. **Model Evaluation**
- Evaluated performance using metrics:
  - **RMSE (Root Mean Squared Error)**  
  - **R² Score**  
- Achieved reliable predictions across both regression and classification modes.

## 📊 Results

**Regression Mode:**  
R² Score: 0.89  
RMSE: 0.42  

**Classification Mode:**  
Accuracy: 0.90  
F1 Score: 0.91  

✅ The Random Forest model provided highly accurate results in both settings, proving effective for predicting movie ratings and sentiment trends.


## 📂 Project Structure
