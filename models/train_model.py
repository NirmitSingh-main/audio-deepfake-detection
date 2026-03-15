import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from data_processing.create_dataset import create_feature_dataset


print("Loading dataset...")

X, y = create_feature_dataset("DATASET")

print("Dataset shape:", X.shape)


# ---------------------------
# Train Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)


# ---------------------------
# SVM MODEL
# ---------------------------

print("\nTraining SVM model...")

svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))

svm.fit(X_train, y_train)

svm_predictions = svm.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)

print("SVM Accuracy:", svm_accuracy)


# ---------------------------
# RANDOM FOREST MODEL
# ---------------------------

print("\nTraining Random Forest model...")

rf = RandomForestClassifier(n_estimators=100, class_weight="balanced")

rf.fit(X_train, y_train)

rf_predictions = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("Random Forest Accuracy:", rf_accuracy)


# ---------------------------
# Save best model
# ---------------------------

joblib.dump(svm, "saved_models/svm_model.pkl")
joblib.dump(rf, "saved_models/rf_model.pkl")

print("\nModels saved in saved_models folder.")