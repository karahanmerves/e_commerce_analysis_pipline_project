# E-Commerce Sales Analysis Pipeline

## Project Overview

This project aims to **analyze and predict e-commerce sales trends** using **machine learning models**. By leveraging **historical sales data**, we aim to build an ML model that helps **businesses optimize inventory management, predict future demand, and enhance sales strategies**.

### Key Goals:
- **Identify sales patterns** from past e-commerce transactions.
- **Segment customers** based on purchasing behavior.
- **Predict future sales trends** using ML models.
- **Optimize decision-making** with data-driven insights.

### Technology Stack:
- **Machine Learning**: Scikit-learn, XGBoost, RandomForest, LinearRegression
- **Data Tracking & Versioning**: **DVC**
- **Experiment Tracking & Model Logging**: **MLflow**
- **Database**: PostgreSQL
- **Code & Pipeline Management**: **Git & GitHub**

---

## Pipeline Architecture & Workflow

### 📂 Project Structure
```
├── .dvc/                    # DVC cache and configuration files
│   ├── cache/
│   ├── tmp/
│   ├── config
│
├── data/
│   ├── raw/                # Raw sales data from PostgreSQL
│   │   ├── .gitignore
│   │   ├── sales_data.csv
│   ├── processed/          # Cleaned and preprocessed data
│       ├── .gitignore
│       ├── train_data.csv
│       ├── test_data.csv
│
├── mlruns/                  # MLflow experiment tracking
│
├── models/                  # Trained model artifacts (not tracked by Git)
│   ├── .gitignore
│   ├── LinearRegression.pkl
│   ├── RandomForest.pkl
│   ├── XGBoost.pkl
│
├── src/
│   ├── __pycache__/
│   ├── analyze_data.ipynb   # Exploratory Data Analysis (EDA)
│   ├── db_connection.py     # Database connection setup
│   ├── evaluate.py          # Model evaluation scripts
│   ├── fetch_data.py        # Fetches data from PostgreSQL
│   ├── preprocess.py        # Data cleaning, feature engineering, outlier removal
│   ├── train.py             # Model training & hyperparameter tuning
│   ├── mlflow_tracking.py   # Logs models & metrics to MLflow
│
├── .dvcignore               # Ignore files for DVC
├── .gitignore               # Ignore files for Git
├── dvc.lock                 # DVC lock file
├── dvc.yaml                 # DVC pipeline definition
├── mlruns.db                # MLflow tracking database
├── params.yaml              # Hyperparameter configurations
├── README.md                # Project documentation
├── requirements.txt         # Project dependencies
```

### End-to-End Workflow:
**1. Data Fetching**: Extracts raw e-commerce transaction data from **PostgreSQL**.  
**2. Data Preprocessing**:
   - Cleans the data and handles missing values.
   - **Removes outliers** using the **IQR method**.
   - Encodes categorical variables and scales numerical features.  
**3. Feature Engineering**: Constructs relevant features for model training.  
**4. Model Training**:
   - Trains **RandomForest, XGBoost, and LinearRegression** models.
   - Uses **hyperparameter tuning** to improve performance.  
**5. Experiment Tracking**: Logs results in **MLflow** for model comparison.  
**6. Pipeline Automation**: Managed using **DVC**, ensuring reproducibility.  

---

## Hyperparameter Optimization & Model Performance

For **XGBoost**, hyperparameters were optimized using **GridSearchCV**:
```yaml
xgboost_learning_rate: 0.2 
xgboost_n_estimators: 200
xgboost_max_depth: 9
```
### Model Performance After Tuning:
| Model                | MAE       | RMSE      |
|----------------------|-----------|-----------|
| **LinearRegression** | 0.667     | 0.911     |
| **RandomForest**     | 0.533     | 0.776     |
| **XGBoost (Tuned)**  | **0.329** | **0.501** |

**XGBoost** outperforms other models in both **MAE (Mean Absolute Error)** and **RMSE (Root Mean Squared Error)**, showing the best predictive accuracy.

---

## Running the Pipeline

### **1️. Clone the Repository & Install Dependencies**
```bash
git clone (https://github.com/karahanmerves/e_commerce_analysis_pipline_project)
cd e_commerce_analysis_pipeline
pip install -r requirements.txt
```

### **2️. Execute the ML Pipeline with DVC**
```bash
dvc repro
```
> This will **fetch the latest data, preprocess it, train the models, and log results in MLflow**.

### **3️. Track Changes with Git & DVC**
```bash
# Track new changes in DVC & Git
dvc commit
git add .
git commit -m "Updated pipeline with outlier removal and hyperparameter tuning"
git push
```

---

## MLflow & Git Integration

**MLflow** automatically logs:
- Model artifacts (trained models)
- Experiment metrics (MAE, RMSE)
- Hyperparameters used in training

**DVC** ensures:
- Versioning of **datasets & preprocessed files**.
- Reproducibility of the **entire ML pipeline**.

**Git** manages:
- **Code & pipeline updates**.
- Version control for **DVC metadata & MLflow experiments**.

### **To View MLflow Experiment Tracking**:
Run the command:
```bash
mlflow ui
```
You can access the MLflow UI by running mlflow ui and navigating to either http://127.0.0.1:5000 or http://localhost:5000, depending on your system configuration.

---

## Key Takeaways
- **DVC + MLflow + Git** ensures a **reproducible, automated ML pipeline**.
- **Hyperparameter tuning** significantly improved **XGBoost** performance.
- **IQR outlier removal** enhanced model generalization.
- **Easily scalable** for future datasets and models.




