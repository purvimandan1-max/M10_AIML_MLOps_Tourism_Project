# for data manipulation
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# for model serialization
import joblib

# for hugging face model upload
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# for experiment tracking
import mlflow

# Use local MLflow store for GitHub Actions
mlflow.set_tracking_uri("file:///tmp/mlruns")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

# Load locally prepared train-test data from data-prep step
DATA_DIR = "tourism_project/data"

print("Files in data directory:", os.listdir(DATA_DIR))

Xtrain = pd.read_csv(os.path.join(DATA_DIR, "Xtrain.csv"))
Xtest  = pd.read_csv(os.path.join(DATA_DIR, "Xtest.csv"))
ytrain = pd.read_csv(os.path.join(DATA_DIR, "ytrain.csv"))
ytest  = pd.read_csv(os.path.join(DATA_DIR, "ytest.csv"))

# Define feature groups
numeric_features = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'PitchSatisfactionScore', 'OwnCar',
    'NumberOfChildrenVisiting', 'MonthlyIncome'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
    'MaritalStatus', 'Designation'
]

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75],
    'xgbclassifier__max_depth': [2, 3],
    'xgbclassifier__learning_rate': [0.05, 0.1]
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "test_accuracy": test_report['accuracy']
    })

    # Save model
    model_path = "best_tourism_failure_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    # Upload model to Hugging Face Model Hub
    repo_id = "kalrap/tourism_failure_model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )

    print("Training completed and model uploaded to Hugging Face.")
