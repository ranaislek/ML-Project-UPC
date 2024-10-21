# Car Auction Price Prediction Project

<img src="https://d3m.upc.edu/en/logosfooter-en/upc/@@images/image" alt="UPC Logo" width="300"/>

## Overview

This project is part of the Master’s in Big Data Management and Analytics (BDMA) program at **Universitat Politècnica de Catalunya (UPC)**. It aims to build machine learning models to predict the final selling prices of used cars at auctions. The project utilizes the "Used Car Auction Prices" dataset from Kaggle, which includes various vehicle and auction attributes. By developing predictive models, the goal is to provide accurate price forecasts and identify key factors that influence car auction prices.

## Team

- **Simon Coessens**  
  Student, BDMA  
  [simon.coessens@estudiantat.upc.edu](mailto:simon.coessens@estudiantat.upc.edu)

- **Rana Islek**  
  Student, BDMA  
  [rana.islek@estudiantat.upc.edu](mailto:rana.islek@estudiantat.upc.edu)

- **Professor**: Marta Arias Vicente

## Project Structure

The repository is organized as follows:

- **data/**: Contains the datasets used for training and analysis.

  - `auction.csv`: Raw auction data.
  - `car_prices.csv`: Additional data on car prices.
  - `dfcar_processed.csv`: Processed dataset used in modeling.
  - `statelatlong.csv`: Geographic data for states.

- **figures/**: Includes visualizations used in data exploration and results analysis.

  - Examples: `distribution_price.png` (price distribution), `sales_by_age.png` (sales trends by car age), `corr_numerical_heatmap.png` (correlation heatmap).

- **models/**: Directory for saved machine learning models.

  - `decision_tree_model.joblib`, `random_forest_model.joblib`, `xgboost_model.joblib`, etc.

- **notebooks/**: Jupyter notebooks for different steps of the project.

  - `01_data_exploration.ipynb`: Data exploration and cleaning.
  - `02_Linear_regression.ipynb`: Linear regression model training.
  - `03_Decision_trees.ipynb`: Training of decision tree models.
  - `04_knn.ipynb`: K-Nearest Neighbors modeling.
  - `05_svm.ipynb`: Support Vector Machine modeling.

- **report/**: Documentation and reports related to the project.
  - `report.pdf`: Final project report summarizing the methodology and results.
  - `Machine_Learning_Project_Proposal.pdf`: Initial project proposal.
  - `ml-project.ipynb`: A consolidated notebook covering the complete project pipeline.

## Main Results

Several machine learning models were evaluated for their ability to predict car auction prices:

- **Linear Regression**:

  - R²: 90.96%
  - Mean Absolute Error (MAE): 0.16

- **Decision Tree Regressor**:

  - R²: 0.91
  - MAE: 0.15

- **K-Nearest Neighbors (KNN)**:

  - R²: 0.92
  - MAE: 0.14

- **Support Vector Machine (SVM)**:
  - R²: 0.79
  - MAE: 0.24
  - Highest MSE among the models evaluated.

The performance comparison is illustrated in the figure below, showing MAE, MSE, RMSE, and R² values for each model.

![Model Comparison](/figures/result.png)

_Figure: Comparison of MAE, MSE, RMSE, and R² for Decision Tree, KNN, SVM, and Linear Regression models._

## Key Insights

- Cars aged 4 years or less had the highest number of transactions.
- Most auctions occurred on weekdays, with peak activity in the early morning.
- Important predictive features included car age, odometer reading, and the state where the auction took place.
- The MMR (Manheim Market Report) value was excluded from the model to prevent information leakage.

## Future Work

Potential improvements and extensions for the project include:

- **Feature Importance Analysis**: It would be valuable to identify which variables have the highest impact on predicting car prices. While we attempted to use the `statsmodel` library with the Linear Regression model to analyze feature importance through model coefficients, this approach resulted in memory errors multiple times. Further exploration with different techniques or optimizations could help resolve this issue.
- **Model Explainability**: Investigating explainability methods for other models, such as Decision Trees, Random Forests, and XGBoost, would provide insights into which features are driving the predictions. Tools like SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) could be employed to better understand the decision-making process of the models.

- **Feature Engineering**: Adding external data sources (e.g., economic indicators or regional trends) may enhance the model's predictive power.
- **Model Optimization**: Further hyperparameter tuning and exploring ensemble techniques could improve model accuracy.
- **Deployment**: Integrating the model into a web application for real-time price prediction would make it accessible for practical use.

---
