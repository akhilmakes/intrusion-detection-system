# src/data_loader.py
import pandas as pd
from src import config

def load_data():
    """Loads training and testing data from paths specified in config."""
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_1_df = pd.read_csv(config.TEST_1_PATH)
    test_2_df = pd.read_csv(config.TEST_2_PATH)
    print("Data loaded successfully.")
    return train_df, test_1_df, test_2_df