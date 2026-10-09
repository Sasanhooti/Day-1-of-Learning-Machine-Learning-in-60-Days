"""Inspect the synthetic dataset without training a model."""

import numpy as np

from data import create_dataset


def analyze_dataset() -> None:
    df = create_dataset()

    feature_columns = ["Area", "Rooms", "Age", "Distance"]
    X = df[feature_columns]
    y = df["Price"]

    print("Dataset preview:")
    print(df.head(), end="\n\n")

    print(f"Samples: {len(df)}")
    print(f"Features: {len(feature_columns)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Missing values in dataset: {int(df.isna().sum().sum())}")

    print("\nSummary statistics:")
    for column in df.select_dtypes(include=np.number).columns:
        values = df[column].to_numpy(dtype=float)
        print(
            f"{column:>10} | "
            f"mean={values.mean():8.2f} | "
            f"min={values.min():8.2f} | "
            f"max={values.max():8.2f}"
        )

    print("\nNote: these are synthetic examples; no model was trained.")


if __name__ == "__main__":
    analyze_dataset()
