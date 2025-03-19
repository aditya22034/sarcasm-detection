# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# # Load the final dataset
# df = pd.read_csv("/kaggle/working/audio_features_WavLM_base.csv")

# # Extract labels
# y = df["Sarcasm"].values  # Labels (0: No sarcasm, 1: Sarcasm)

# # Extract context features (from csv1_)
# X_context = df[[col for col in df.columns if col.startswith("audio_c_feature_")]].values

# # Extract utterance features (from csv2_)
# X_utterance = df[[col for col in df.columns if col.startswith("audio_u_feature_")]].values

# # Convert to NumPy arrays
# X_context = np.array(X_context, dtype=np.float32)
# X_utterance = np.array(X_utterance, dtype=np.float32)
# y = np.array(y, dtype=np.float32)

# # Train-test split (80%-20%)
# Xc_train, Xc_test, Xu_train, Xu_test, y_train, y_test = train_test_split(
#     X_context, X_utterance, y, test_size=0.2, random_state=42, stratify=y
# )

# # CNN Model for Sarcasm Detection
# input_dim = 768  # Number of features per input

# # Context Branch
# input_context = keras.Input(shape=(input_dim,))
# context_branch = layers.Reshape((input_dim, 1))(input_context)
# context_branch = layers.Conv1D(filters=64, kernel_size=3, activation="relu")(context_branch)
# context_branch = layers.MaxPooling1D(pool_size=2)(context_branch)
# context_branch = layers.Flatten()(context_branch)

# # Utterance Branch
# input_utterance = keras.Input(shape=(input_dim,))
# utterance_branch = layers.Reshape((input_dim, 1))(input_utterance)
# utterance_branch = layers.Conv1D(filters=64, kernel_size=3, activation="relu")(utterance_branch)
# utterance_branch = layers.MaxPooling1D(pool_size=2)(utterance_branch)
# utterance_branch = layers.Flatten()(utterance_branch)

# # Concatenation
# merged = layers.Concatenate()([context_branch, utterance_branch])
# merged = layers.Dense(32, activation="relu")(merged)
# #merged = layers.Dropout(0.2)(merged)
# #merged = layers.Dense(64, activation="relu")(merged)
# output = layers.Dense(1, activation="sigmoid")(merged)  # Sigmoid for binary classification

# # Define Model
# model = keras.Model(inputs=[input_context, input_utterance], outputs=output)
# model.summary()

# # Compile Model
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# # Train Model
# model.fit([Xc_train, Xu_train], y_train, epochs=10, batch_size=32, validation_data=([Xc_test, Xu_test], y_test))

# # Predictions
# y_train_pred = (model.predict([Xc_train, Xu_train]) > 0.5).astype(int)
# y_test_pred = (model.predict([Xc_test, Xu_test]) > 0.5).astype(int)

# # Print Classification Reports
# print("Train Set Classification Report:\n", classification_report(y_train, y_train_pred))
# print("Test Set Classification Report:\n", classification_report(y_test, y_test_pred))

# """
# MERGING THE AUDIO EMBEDDINGS OF CONTEXT AND UTTERNACE WITH LABELS AND OTHER FEATURES
# MERGING LanguageBind Embeddings
# """

import pandas as pd

# Load the CSV files
csv1 = pd.read_csv("C:/Users/lenovo/Downloads/xlsr_context_features.csv")
csv2 = pd.read_csv("C:/Users/lenovo/Downloads/xlsr_utterance_features.csv")
map_df = pd.read_csv("C:/Users/lenovo/Downloads/context_to_utterance_map.csv")

# Remove the 'audio_context/' and 'audio_utterance/' prefixes from map.csv
map_df["audio_context"] = map_df["audio_context"].str.replace("audio_context/", "", regex=False)
map_df["audio_utterance"] = map_df["audio_utterance"].str.replace("audio_utterance/", "", regex=False)

# Extract features (excluding the first column which is file_name)
features_csv1 = csv1.iloc[:, 1:].copy()  # Features from csv1
features_csv2 = csv2.iloc[:, 1:].copy()  # Features from csv2

# Rename columns to distinguish between csv1 and csv2 features
features_csv1.columns = [f"audio_c_feature_{col}" for col in features_csv1.columns]
features_csv2.columns = [f"audio_u_feature_{col}" for col in features_csv2.columns]

# Add file_name back to features for merging
features_csv1.insert(0, "filename", csv1.iloc[:, 0])
features_csv2.insert(0, "filename", csv2.iloc[:, 0])

# Merge csv1 with map.csv using audio_context (which is file_name in csv1)
merged_df = map_df.merge(features_csv1, left_on="audio_context", right_on="filename", how="inner")

# Merge csv2 with the updated dataframe using audio_utterance (which is file_name in csv2)
merged_df = merged_df.merge(features_csv2, left_on="audio_utterance", right_on="filename", how="inner", suffixes=("_csv1", "_csv2"))

# Drop redundant filename columns from csv1 and csv2
merged_df.drop(columns=["filename_csv1", "filename_csv2"], inplace=True)

# Rename columns to keep them organized
#merged_df.rename(columns={"audio_context": "file_csv1", "audio_utterance": "file_csv2"}, inplace=True)

# Save the final dataset
merged_df.to_csv("audio_features_xlsr.csv", index=False)

print("Merged dataset saved.")

