import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load dataset
df = pd.read_csv("/content/audio_features_lb.csv") # just change the path of the file for different embedings from different pretrained models

# Extract labels
y = df["Sarcasm"].values  # 0 = No sarcasm, 1 = Sarcasm

# Extract features
X_context = df[[col for col in df.columns if col.startswith("audio_c_feature_")]].values
X_utterance = df[[col for col in df.columns if col.startswith("audio_u_feature_")]].values

# Normalize the features
scaler = StandardScaler()
X_context = scaler.fit_transform(X_context)
X_utterance = scaler.fit_transform(X_utterance)

# Convert to NumPy arrays
X_context = np.array(X_context, dtype=np.float32)
X_utterance = np.array(X_utterance, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Train-test split
Xc_train, Xc_test, Xu_train, Xu_test, y_train, y_test = train_test_split(
    X_context, X_utterance, y, test_size=0.2, random_state=42, stratify=y
)

# Dynamically set input dimensions
input_dim = Xc_train.shape[1]

# Context Branch (Fully Connected)
input_context = keras.Input(shape=(input_dim,))
context_branch = layers.Dense(256, activation="relu")(input_context)
context_branch = layers.BatchNormalization()(context_branch)
context_branch = layers.Dense(128, activation="relu")(context_branch)
context_branch = layers.Dropout(0.3)(context_branch)

# Utterance Branch (Fully Connected)
input_utterance = keras.Input(shape=(input_dim,))
utterance_branch = layers.Dense(256, activation="relu")(input_utterance)
utterance_branch = layers.BatchNormalization()(utterance_branch)
utterance_branch = layers.Dense(128, activation="relu")(utterance_branch)
utterance_branch = layers.Dropout(0.3)(utterance_branch)

# Merge both branches
merged = layers.Concatenate()([context_branch, utterance_branch])
merged = layers.Dense(128, activation="relu")(merged)
merged = layers.Dropout(0.3)(merged)
merged = layers.Dense(64, activation="relu")(merged)
merged = layers.Dropout(0.2)(merged)
output = layers.Dense(1, activation="sigmoid")(merged)  # Binary classification

# Define Model
model = keras.Model(inputs=[input_context, input_utterance], outputs=output)
model.summary()

# Compile Model with Lower Learning Rate
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Train Model
model.fit([Xc_train, Xu_train], y_train, epochs=20, batch_size=32, validation_data=([Xc_test, Xu_test], y_test))

# Predictions
y_train_pred = (model.predict([Xc_train, Xu_train]) > 0.5).astype(int)
y_test_pred = (model.predict([Xc_test, Xu_test]) > 0.5).astype(int)

# Print Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
print("\n✅ Train Accuracy:", train_acc)
print("✅ Test Accuracy:", test_acc)

# Print Classification Reports
print("\nTrain Set Classification Report:\n", classification_report(y_train, y_train_pred))
print("Test Set Classification Report:\n", classification_report(y_test, y_test_pred))
