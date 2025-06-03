# Regression: Cardiovascular Disease Prediction

## ℹ️ Introduction

According to the World Health Organization, heart disease is responsible for an estimated 12 million deaths globally each year. In the US and other developed countries, cardiovascular diseases accounts for nearly half of all deaths. Early detection and prognosis of these conditions can play a critical role in guiding lifestyle changes for high-risk individuals in order to prevent complications. This project aims to identify the most significant risk factors contributing to coronary heart disease (CHD) and predict overall risk using logistic regression analysis.

## 🎯 Project Objectives

- Data Exploration and Understanding

- Data Preprocessing

- Logistic Regression Model Development and Optimization

- Model Evaluation and Threshold Optimization

- Model Interpretation and Insights

- Reporting and Presentation

## 🛠️ Libraries Used
* ![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
* 🐼 Pandas
* 🔢 NumPy
* 📊 Matplotlib
* 🎨 Seaborn
* 🤖 Scikit-learn
* ⚖️ Imbalanced-learn
* 📓 Jupyter Notebook

## 💾 Dataset

The analysis is based on a dataset (source: [Kaggle Dataset](https://www.kaggle.com/datasets/christofel04/cardiovascular-study-dataset-predict-heart-disea)) containing various demographic, behavioral, and medical features of individuals. These features include:
* **Demographic:** `age`, `sex`, `education`
* **Behavioral:** `is_smoking`, `cigsPerDay` (cigarettes per day)
* **Medical History:** `BPMeds` (on blood pressure medication), `prevalentStroke`, `prevalentHyp` (prevalent hypertension), `diabetes`
* **Physical Examination:** `totChol` (total cholesterol), `sysBP` (systolic blood pressure), `diaBP` (diastolic blood pressure), `BMI` (Body Mass Index), `heartRate`, `glucose`
* **Target Variable:** `TenYearCHD` (binary: 0 for no CHD in 10 years, 1 for CHD in 10 years)

## ⚙️ Project Workflow

1.  **Introduction:** Outlines project objectives and data.
2.  **Data Cleaning:** Initial data loading, inspection, handling of irrelevant columns, check for duplicates and initial assessment of missing values.
3.  **Exploratory Data Analysis (EDA):**
    * Analysis of the target variable (`TenYearCHD`) distribution, noting class imbalance.
    * Analysis of numerical features (histograms, box plots) to understand distributions and identify outliers.
    * Analysis of categorical features (count plots) to understand frequencies.
    * Analysis to explore relationships between predictor variables and the `TenYearCHD` target (numerical vs. target using box plots; categorical vs. target using count plots with hue).
    * Correlation analysis among numerical features (heatmap) to identify multicollinearity.
4.  **Data Preprocessing:**
    * Train-Test Split (80/20 split, stratified by `TenYearCHD`).
    * Feature Selection: `diaBP` was dropped due to high correlation with `sysBP`.
    * Development of transformation pipelines for numerical features (`SimpleImputer` with median strategy, `RobustScaler`) and categorical features (`SimpleImputer` with most frequent strategy, `OneHotEncoder` with `drop='first'`).
    * Application of these transformations using `ColumnTransformer`.
5.  **Model Development:**
    * The primary model used is **Logistic Regression**.
    * **SMOTE (Synthetic Minority Over-sampling Technique)** with `k_neighbors=4` was applied to the training data to address class imbalance.
    * Hyperparameter tuning for the Logistic Regression's `C` parameter was performed using **`GridSearchCV`** with 5-fold cross-validation, optimizing for ROC AUC (best `C` found was 0.01).
6.  **Model Evaluation:**
    * The final tuned model was evaluated on the unseen test set.
    * The **Precision-Recall curve** was analyzed to select an optimal probability threshold, prioritizing **Recall** for the CHD positive class while considering the F1-score.
    * Performance was assessed using a Confusion Matrix, Accuracy, Precision, Recall, F1-score, and ROC AUC score at the chosen optimal threshold.
7.  **Model Interpretation:**
    * Coefficients and Odds Ratios from the final logistic regression model were extracted and analyzed to understand the impact of each predictor on CHD risk.
8.  **Conclusion:** Actionable insights, limitations of the model and further improvements.

## 📊 Key Results & Insights

* The EDA confirmed class imbalance (~15% CHD cases) and identified key relationships and data characteristics.
* The final Logistic Regression model, after SMOTE (k=4) and C-parameter tuning (C=0.01), achieved the following approximate performance on the test set at an optimal threshold of **0.4687**:
    * **ROC AUC:** ~0.70
    * **Recall (Sensitivity for CHD):** ~0.7353 (Successfully identified about 73.5% of actual CHD cases)
    * **Precision (for CHD):** ~0.2475
    * **F1-score (for CHD):** ~0.3704
    * **Accuracy:** ~0.6239
* **Key Predictors for Increased CHD Risk (based on Odds Ratios):** `age`, `cigsPerDay`, `sysBP`, and being `male` were among the most significant factors.
* **Protective Factors:** Higher levels of `education` were associated with lower odds of CHD.
* **Discussion:** The model shows a good ability to identify a majority of CHD cases (high recall), which is crucial in a clinical context. However, the precision is modest, indicating a fair number of false positives.

## 🚀 How to Use/Run

1.  **Clone the repository.**
2.  **Ensure you have Python installed.**
3.  **Install necessary libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
    ```
4.  **Open and run the Jupyter Notebook:**
    * `cardio.ipynb` contains all the analysis.
5.  **Data:** The `train.csv` should be in the same directory or an appropriate path specified in the notebook.

## ⚠️ Limitations

1. **Correlation vs. Causation**: This model identifies statistical links between predictors and CHD risk. However, these associations do not necessarily imply causal relationships.

2. **Generalizability**: The model itself was trained and evaluated on a specific dataset. The model's performance on differing populations or geographical settings may vary.

3. **Performance**: Recall of **73.5%** is a positive outcome, but Precision of **24.75%** is low, resulting in a modest F1-score of **~0.37**. This results in a high rate of false positives, which is a problem in a clinical setting without follow-up procedures. The ability to discriminate between classes, which is measured by ROC AUC, is **~0.70**, this is a fair result but suggests room for improvement.

4. **Data Limitations**:
    - The dataset size of **3390** samples is moderate and might limit the ability to detect more complex interactions effectively.

    - The dataset itself might not have all the relevant risk factors for CHD. Examples like dietary habits, family health history and physical activity levels are all missing. Inclusion of such information could lead to further insights and improve model performance.

    - Since the dataset had missing data, imputation was used to fill the blanks. This can introduce some level of bias.

5. **Model Simplicity**: Logistic regression is a linear model, which may not capture complex non-linear relationships between features.

## 💡 Further Improvements

- **Data Enrichment**: If possible, incorporating more data with relevant features known to be associated with increased CHD risk could significantly enhance predictive power.

- **Weak Coefficients**: Some features showed surprisingly weak effects in this model. Further investigation could be done that explores interactions more deeply and considers if their impact is mediated by other variables in the model.

- **Other Models**: Exploring non-linear models could capture more complex relationships and lead to improved performance metrics.


## 🏁 Conclusion Summary

This project successfully developed a regularized logistic regression model with SMOTE for 10-year CHD risk prediction. Key risk factors were identified, and the model was optimized to achieve high recall, acknowledging the trade-off with precision. While the model provides useful insights and fair discriminative ability, its practical application would require careful consideration of its false positive rate.

---

## 👤 Author

* Laisvis Remeikis
* 🔗 [LinkedIn](https://www.linkedin.com/in/laisvis-remeikis/)
