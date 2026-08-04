import matplotlib.pyplot as plt

# Model names
models = [
    "Linear\nRegression",
    "Decision\nTree",
    "Random\nForest",
    "SVR",
    "ANN"
]

# RMSE values
rmse = [
    31327.80,
    41784.63,
    28480.78,
    88653.09,
    40558.51
]

# Create figure
plt.figure(figsize=(8,5))

# Draw bars
bars = plt.bar(models, rmse)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 1000,
        f"{height:.2f}",
        ha='center',
        fontsize=9
    )

# Title and labels
plt.title("Figure 6.1 - Model Comparison (RMSE)")
plt.xlabel("Machine Learning Models")
plt.ylabel("RMSE")

# Save figure
plt.savefig("Figure_6_1_Model_Comparison_RMSE.png", dpi=300, bbox_inches="tight")

# Show chart
plt.show()

