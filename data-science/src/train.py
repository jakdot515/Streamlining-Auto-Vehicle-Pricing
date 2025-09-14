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
from matplotlib import pyplot as plt

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    
    # -------- WRITE YOUR CODE HERE --------
    
    # Step 1: Define arguments for train data, test data, model output, and RandomForest hyperparameters. Specify their types and defaults.  
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument('--criterion', type=str, default='gini',
                        help='The function to measure the quality of a split')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples.')

    args = parser.parse_args()

    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # -------- WRITE YOUR CODE HERE --------

    # Step 2: Read the train and test datasets from the provided paths using pandas. Replace '_______' with appropriate file paths and methods.  
    train_df = pd.read_csv(Path(args.train_data)/"train.csv")
    test_df = pd.read_csv(Path(args.test_data)/"test.csv")
    
    # Step 3: Split the data into features (X) and target (y) for both train and test datasets. Specify the target column name.
    y_train = train_df['Failure']
    X_train = train_df.drop(columns=['Failure'])
    y_test = test_df['Failure']
    X_test = test_df.drop(columns=['Failure'])

    # Step 4: Initialize the RandomForest Regressor with specified hyperparameters, and train the model using the training data.
    model = DecisionTreeClassifier(criterion=args.criterion, max_depth=args.max_depth)
    model.fit(X_train, y_train)
    
    # Step 5: Log model hyperparameters like 'n_estimators' and 'max_depth' for tracking purposes in MLflow.
    mlflow.log_param("model", "DecisionTreeClassifier")
    mlflow.log_param("criterion", args.criterion)
    mlflow.log_param("max_depth", args.max_depth)
    
    # Step 6: Predict target values on the test dataset using the trained model, and calculate the mean squared error.
    yhat_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, yhat_test)
    print(f'Accuracy of Decision Tree classifier on test set: {accuracy:.2f}')
    # Step 7: Log the MSE metric in MLflow for model evaluation, and save the trained model to the specified output path.
    mlflow.log_metric("Accuracy", float(accuracy))
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

if __name__ == "__main__":
    
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()

