import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ======================================
# LOAD CLEANED DATA
# ======================================

df = pd.read_csv("hypertension_cleaned.csv")

print("Dataset shape:", df.shape)

# ======================================
# DEFINE TARGET
# ======================================

target_col = "has_hypertension"

X = df.drop(target_col, axis=1)
y = df[target_col]

# ======================================
# TRAIN TEST SPLIT
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

# ======================================
# SCALE FEATURES
# ======================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================================
# TRAIN MODEL
# ======================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ======================================
# EVALUATE MODEL
# ======================================

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ======================================
# SAVE MODEL + SCALER
# ======================================

if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(model, "models/hypertension_model.pkl")
joblib.dump(scaler, "models/hypertension_scaler.pkl")

print("\n✅ Model and scaler saved successfully!")