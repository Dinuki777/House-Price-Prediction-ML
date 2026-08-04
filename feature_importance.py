import joblib
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load("house_price_model.pkl")

# -----------------------------
# Feature Names
# -----------------------------
feature_names = [
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

# -----------------------------
# Get Random Forest Model
# -----------------------------
rf_model = model.named_steps["model"]

# -----------------------------
# Feature Importance
# -----------------------------
importance = rf_model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

# Sort values
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(9,6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Figure 6.3 - Feature Importance (Random Forest)")

plt.gca().invert_yaxis()

plt.grid(axis="x", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "Figure_6_3_Feature_Importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure_6_3_Feature_Importance.png saved successfully!")