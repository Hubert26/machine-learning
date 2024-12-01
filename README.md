# Lending Club Loan Default Prediction

This project focuses on building a classification model that predicts whether a potential borrower will repay their loan. The dataset used is from Lending Club, a peer-to-peer lending company, which connects borrowers with investors via an online platform. Lending Club provides data on loans issued between 2007 and 2011, including details about whether each loan was fully paid or charged off.

## Objective

The goal of this project is to create a predictive model that estimates the likelihood of loan repayment based on various factors. The dataset includes information such as credit scores, loan amount, income, employment history, and more. The primary task is to build a classification model to predict whether a borrower will fully repay their loan or default (charged off).

## Dataset

The dataset contains loan details, including the loan status, which indicates whether the loan was fully paid or charged off. Other important features include:

- FICO Score
- Loan amount
- Income
- Employment history
- Credit history
- Mortgage status
- Loan term, etc.

The dataset `Loan_data.csv` is accompanied by a file `LCDataDictionary.csv` that describes each feature.

## Project Stages

### 1. Data Processing
The data processing step involved cleaning and preprocessing the dataset. This included handling missing values, converting categorical variables to numerical ones, and scaling features to prepare the data for modeling.

### 2. Exploratory Data Analysis (EDA)
In this stage, I performed a detailed analysis of the dataset to uncover patterns and insights. Key questions explored included:

- How the FICO score is related to loan repayment likelihood.
- The relationship between the credit history and default probability.
- How factors like annual income, loan amount, and mortgage status impact the likelihood of loan default.
- The potential correlation between the requested loan size and repayment probability.

### 3. Feature Engineering
I created 20 new features based on the existing data. These features aimed to provide additional insights into the borrower’s financial situation, such as ratios between income and loan amount, and categorization of credit scores into ranges.

### 4. Modeling

#### 4.1. Clustering
I applied various clustering techniques (including K-Means, DBSCAN, and hierarchical clustering) to identify distinct segments of borrowers. The optimal number of clusters was determined using methods like the Elbow Method and Silhouette Analysis. The results showed the following:

- **Silhouette Score**: Values from both K-Means and Hierarchical Clustering were around 0.15, indicating weak clustering with some overlap between the clusters.
- **DBSCAN**: This algorithm classified all samples as outliers, suggesting that the dataset might be difficult to cluster effectively.

#### 4.2. Model Training
I trained five different models using various algorithms, including:

- K-nearest neighbors (KNN)
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machines (SVM)

For each model, I evaluated performance using the AUROC (Area Under the Receiver Operating Characteristic Curve) score.

#### 4.3. PCA and Model Comparison
I reduced the dimensionality of the data using Principal Component Analysis (PCA) and evaluated how the previously trained models performed on the compressed dataset. The results showed a noticeable drop in performance for all models after PCA, which is expected as PCA reduces the number of features at the cost of some information loss. The AUROC scores were compared to the models trained on the original data.

#### 4.4. Final Model
The final model was selected based on the best AUROC score. I performed cross-validation, hyperparameter tuning, and ensured class balancing to build the most accurate model. The final model achieved an AUROC score of over 80%, meeting the project requirements.

### Conclusion

This project involved a comprehensive process of data exploration, feature engineering, clustering, and model evaluation. Key insights include:

- **Clustering**: K-Means and Hierarchical Clustering produced similar results with low silhouette scores (~0.15), suggesting weak separation of data. DBSCAN identified all samples as outliers, which implies difficulty in clustering the dataset effectively.
- **Classification**: No significant differences were found between standardizing numerical data versus applying one-hot encoding and standardization. However, dimensionality reduction via PCA caused a drop in performance across all models.
- **Final Model**: The XGBoost model, after cross-validation and tuning, performed consistently well with an AUROC score above 80%, meeting the project requirements.

The steps followed in this project showcase the practical application of data science techniques in building a predictive model for financial data.

## Technologies Used

- **Python**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Clustering Algorithms**: K-Means, DBSCAN, Hierarchical Clustering
- **Machine Learning Algorithms**: KNN, Logistic Regression, Random Forest, Gradient Boosting, SVM
- **Dimensionality Reduction**: Principal Component Analysis (PCA)
- **Evaluation Metric**: AUROC Score

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Hubert26/machine-learning/blob/main/LICENSE.txt) file for details.
