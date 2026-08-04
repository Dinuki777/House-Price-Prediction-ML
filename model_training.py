import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("train.csv")

print(df.head())
print(df.shape)

# -----------------------------
# Selected Features
# -----------------------------
selected_features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "FullBath",
    "TotRmsAbvGrd",
    "YearBuilt",
    "YearRemodAdd"
]

X = df[selected_features]
y = df["SalePrice"]

# -----------------------------
# Identify Column Types
# -----------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "string"]).columns

# -----------------------------
# Preprocessing
# -----------------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -----------------------------
# Models
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Support Vector Regression": SVR(),

    "Neural Network": MLPRegressor(
        hidden_layer_sizes=(100,),
        max_iter=1000,
        random_state=42
    )
}

best_model = None
best_score = float("inf")

print("\n==============================")
print("Model Results")
print("==============================")

# -----------------------------
# Train Models
# -----------------------------
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"{name}: RMSE = {rmse:.2f}")

    if rmse < best_score:
        best_score = rmse
        best_model = pipeline

print("==============================")
print(f"Best RMSE : {best_score:.2f}")
print("==============================")

# -----------------------------
# Save Best Model
# -----------------------------
joblib.dump(best_model, "house_price_model.pkl")

print("Model saved as house_price_model.pkl")

# -----------------------------
# Create Kaggle Submission
# -----------------------------
print("\nCreating Kaggle submission...")

# Load Kaggle test dataset
test_df = pd.read_csv("test.csv")

# Load sample submission
submission = pd.read_csv("sample_submission.csv")

# Select same features
X_submission = test_df[selected_features]

# Predict
submission["SalePrice"] = best_model.predict(X_submission)

# Save submission
submission.to_csv("submission.csv", index=False)

print("submission.csv created successfully!")
print("Ready to upload to Kaggle.")