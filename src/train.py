# src/train.py
import joblib
from src import config, data_loader, preprocess, model

# 1. Load Data
train_df, _, _ = data_loader.load_data()
y_train = train_df['label'].values.reshape(-1, 1)

# 2. Preprocess Data
X_train_processed, training_mean, eigen_vectors_k = preprocess.preprocess_data(train_df)
print(f"Training data shape after PCA: {X_train_processed.shape}")

# 3. Initialize and Train Model
input_size = X_train_processed.shape[1]
hidden_size = int(input_size * 1.5)
mlp_model = model.MLP(input_size=input_size, hidden_size=hidden_size, output_size=1)
mlp_model.train(X_train_processed, y_train, epochs=config.EPOCHS, learning_rate=config.LEARNING_RATE)
print("Model training complete.")

# 4. Save the Model and Preprocessor
joblib.dump(mlp_model, config.MODEL_PATH)
preprocessor_params = {'mean': training_mean, 'pca_vecs': eigen_vectors_k}
joblib.dump(preprocessor_params, config.PREPROCESSOR_PATH)
print(f"Model and preprocessor saved to {config.MODEL_PATH} and {config.PREPROCESSOR_PATH}")