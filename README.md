
 FUTURE SALES PREDICTION
=============================

This project predicts future store sales using historical data from multiple sources,
including store details, oil prices, and holiday events. The notebook is designed in
Google Colab for simplicity, automation, and clear visualization.

---------------------------------------------------
 DATASET DETAILS
---------------------------------------------------
The dataset includes the following CSV files:

1. train.csv          → Contains sales transactions with dates, stores, and items
2. stores.csv         → Information about each store (type, cluster, etc.)
3. oil.csv            → Daily oil prices affecting the economy
4. holidays_events.csv → National and local holidays/events that may influence sales

Dataset Path:
  /content/drive/MyDrive/Colab Notebooks/FutureSalesPrediction/Dataset/

---------------------------------------------------
 LIBRARIES AND INSTALLATION
---------------------------------------------------
Before running the notebook, install the following libraries:

!pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm

---------------------------------------------------
 MODELS USED
---------------------------------------------------
The notebook trains and evaluates the following regression models:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor
- Gradient Boosting Regressor

---------------------------------------------------
 DATA PREPROCESSING STEPS
---------------------------------------------------
1. Handle missing values.
2. Merge datasets into a single DataFrame.
3. Apply encoding for categorical variables.
4. Scale numerical features using StandardScaler.
5. Split data into training and validation sets.

---------------------------------------------------
 MODEL EVALUATION
---------------------------------------------------
Each model is evaluated using the following metrics:
- RMSE (Root Mean Squared Error)
- R² (R-squared Score)

The model with the highest R² and lowest RMSE is considered the best.

---------------------------------------------------
 FEATURE IMPORTANCE & INSIGHTS
---------------------------------------------------
After training, feature importance is analyzed to find the top influential features.

Only Top 5 features are used for the final insights:
1. onpromotion
2. type_enc
3. family_enc
4. store_nbr
5. id

