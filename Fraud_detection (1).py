
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers
import os
import requests

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DATA_PATH = "creditcard.csv"
if not os.path.exists(DATA_PATH):
    print("Downloading credit card fraud dataset...")
    r = requests.get(DATA_URL)
    with open(DATA_PATH, "wb") as f:
        f.write(r.content)
    print("Download complete.")
else:
    print("Dataset already exists.")


df = pd.read_csv(DATA_PATH)
print("Data loaded. Shape:", df.shape)

# --- Data Exploration & Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution (0: Not Fraud, 1: Fraud)')
plt.show()
print('Class counts:')
print(df['Class'].value_counts())

# Feature distribution for a few features
sample_features = ['V1', 'V2', 'V3', 'Amount']
df[sample_features].hist(bins=30, figsize=(10,6))
plt.suptitle('Feature Distributions')
plt.show()

# Correlation heatmap (top features with Class)
correlations = df.corr()['Class'].abs().sort_values(ascending=False)
top_corr_features = correlations[1:6].index.tolist()  # Exclude 'Class' itself
plt.figure(figsize=(8,5))
sns.heatmap(df[top_corr_features + ['Class']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Top Features vs Class)')
plt.show()

X = df.drop(["Class"], axis=1).values
y = df["Class"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a simple neural network
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
print("Training the model...")
model.fit(X_train, y_train, epochs=5, batch_size=2048, validation_split=0.1, verbose=1)

# Evaluate
print("Evaluating on test set...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {accuracy:.4f}")

# Predict and print classification report
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
