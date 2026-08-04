import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

quality = df["OverallQual"].value_counts().sort_index()

plt.figure(figsize=(8,5))
plt.bar(quality.index.astype(str), quality.values)

plt.title("Distribution of Houses by Overall Quality")
plt.xlabel("Overall Quality")
plt.ylabel("Number of Houses")

plt.tight_layout()
plt.savefig("Figure4_1_OverallQuality.png", dpi=300)
plt.show()