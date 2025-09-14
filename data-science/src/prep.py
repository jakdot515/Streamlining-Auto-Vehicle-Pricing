# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")
    return parser.parse_args()


def main(args):
    """Read, preprocess, split, and save datasets"""

    # Reading Data
    df = pd.read_csv(args.raw_data)
    print("✅ Loaded data with shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Step 1: Encode categorical column(s) if present
    cat_col = None
    for candidate in ["Segment", "Type"]:
        if candidate in df.columns:
            cat_col = candidate
            break

    if cat_col:
        print(f"Encoding categorical column: {cat_col}")
        le = LabelEncoder()
        df[cat_col] = le.fit_transform(df[cat_col])
    else:
        print("⚠️ No categorical column ['Segment', 'Type'] found. Skipping encoding.")

    # Step 2: Train/test split
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)
    print(f"Split data → Train: {train_df.shape}, Test: {test_df.shape}")

    # Step 3: Ensure output directories exist
    #os.makedirs(args.train_data, exist_ok=True)
    #os.makedirs(args.test_data, exist_ok=True)

    # Step 4: Save train/test sets
    train_output_path = os.path.join(args.train_data, "train.csv")
    test_output_path = os.path.join(args.test_data, "test.csv")

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    # Step 5: Log metrics
    mlflow.log_metric("train_size", train_df.shape[0])
    mlflow.log_metric("test_size", test_df.shape[0])


if __name__ == "__main__":
    mlflow.start_run()

    args = parse_args()

    print(f"Raw data path: {args.raw_data}")
    print(f"Train dataset output path: {args.train_data}")
    print(f"Test dataset path: {args.test_data}")
    print(f"Test-train ratio: {args.test_train_ratio}")

    main(args)

    mlflow.end_run()
