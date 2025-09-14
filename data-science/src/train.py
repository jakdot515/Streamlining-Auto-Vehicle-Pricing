# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")

    # Arguments for train/test data, model output, and hyperparameters
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--model_output", type=str, help="Path to save trained model")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the trees")
    parser.add_argument("--criterion", type=str, default="mse", help="Function to measure split quality (mse, mae)")

    args = parser.parse_args()
    return args

def main(args):
    '''Read train/test datasets, train RandomForestRegressor, evaluate, log metrics, save model'''

    # Load datasets
    train_df = pd.read_csv(Path(args.train_data)/"train.csv")
    test_df = pd.read_csv(Path(args.test_data)/"test.csv")

    # Split features and target
    y_train = train_df['price']
    X_train = train_df.drop(columns=['price'])
    y_test = test_df['price']
    X_test = test_df.drop(columns=['price'])

    # Map user-friendly criterion to sklearn supported
    criterion_map = {
        "mse": "squared_error",
        "mae": "absolute_error"
    }
    criterion = criterion_map.get(args.criterion, args.criterion)

    # Initialize and train model
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        criterion=criterion,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Log hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("criterion", criterion)

    # Predict and evaluate
    yhat_test = model.predict(X_test)
    mse = mean_squared_error(y_test, yhat_test)
    print(f"RandomForestRegressor Test MSE: {mse:.4f}")

    # Log metrics and save model
    mlflow.log_metric("MSE", float(mse))
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

if __name__ == "__main__":

    with mlflow.start_run():
        # Parse arguments
        args = parse_args()

        print(f"Train dataset input path: {args.train_data}")
        print(f"Test dataset input path: {args.test_data}")
        print(f"Model output path: {args.model_output}")
        print(f"Number of Estimators: {args.n_estimators}")
        print(f"Max Depth: {args.max_depth}")
        print(f"Criterion: {args.criterion}")

        main(args)
