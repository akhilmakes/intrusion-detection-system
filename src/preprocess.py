# src/preprocess.py
import numpy as np
import pandas as pd
from src import config

def preprocess_data(df, training_mean=None, eigen_vectors_k=None):
    """Preprocesses a single dataframe."""
    # Encode categorical variables
    for col in config.CATEGORICAL_COLS:
        unique_values = df[col].unique()
        mapping = {v: i for i, v in enumerate(unique_values)}
        df[col] = df[col].map(mapping)

    # Separate features
    features = df.select_dtypes(include=['float64', 'int64'])
    if 'label' in features.columns:
        features = features.drop(columns=['label'])

    # Normalize features
    for col in features.columns:
        min_val, max_val = features[col].min(), features[col].max()
        if max_val - min_val != 0:
            features[col] = (features[col] - min_val) / (max_val - min_val)
        else:
            features[col] = 0

    # Perform PCA
    if training_mean is None and eigen_vectors_k is None:
        # This block runs for training data
        training_mean = np.mean(features, axis=0)
        mean_sub_features = features - training_mean
        cov_matrix = np.cov(mean_sub_features, rowvar=False)
        eig_val, eig_vec = np.linalg.eigh(cov_matrix)
        
        sort_index = np.argsort(eig_val)[::-1]
        eig_val, eig_vec = eig_val[sort_index], eig_vec[:, sort_index]
        
        cumulative_variance = np.cumsum(eig_val) / np.sum(eig_val)
        k = np.argmax(cumulative_variance >= config.PCA_TARGET_VARIANCE) + 1
        eigen_vectors_k = eig_vec[:, :k]
        
        features_reduced = np.dot(mean_sub_features, eigen_vectors_k)
        return features_reduced, training_mean, eigen_vectors_k
    else:
        # This block runs for test data, using params from training
        mean_sub_test = features - training_mean
        features_reduced = np.dot(mean_sub_test, eigen_vectors_k)
        return features_reduced