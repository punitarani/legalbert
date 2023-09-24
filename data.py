"""data.py"""

import pandas as pd


def load_google_data(train_test: str = "train") -> pd.DataFrame:
    """
    Load Google's Patent Phrase similarity dataset
    """

    if train_test == "train":
        data = pd.read_csv("data/google/train.csv")
    elif train_test == "test":
        data = pd.read_csv("data/google/test.csv")
    elif train_test == "validation":
        data = pd.read_csv("data/google/validation.csv")
    else:
        raise ValueError("train_test must be 'train', 'test', or 'validation'")

    return data
