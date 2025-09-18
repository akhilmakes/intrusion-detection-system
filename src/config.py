# src/config.py

# File paths
TRAIN_PATH = '../data/UNSWNB15_training_coursework.csv'
TEST_1_PATH = '../data/UNSWNB15_testing1_coursework.csv'
TEST_2_PATH = '../data/UNSWNB15_testing2_coursework_no_label.csv'

# Preprocessing settings
CATEGORICAL_COLS = ['proto', 'service', 'state']
PCA_TARGET_VARIANCE = 0.97

# Model training hyperparameters
EPOCHS = 1000
LEARNING_RATE = 0.5
RANDOM_SEED = 20

# Saved model paths
MODEL_PATH = '../saved_models/mlp_model.pkl'
PREPROCESSOR_PATH = '../saved_models/preprocessor.pkl'