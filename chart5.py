import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

top = df["Neighborhood"].value_counts().head(10)

plt.figure(figsize=(10,5))
plt.bar(top.index, top.values)

plt.title("Top 10 Neighborhoods")
plt.xlabel("Neighborhood")
plt.ylabel("Number of Houses")

plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("Figure4_5_Neighborhood.png", dpi=300)
plt.show()