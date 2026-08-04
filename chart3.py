import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

year = df["YearBuilt"].value_counts().sort_index()

plt.figure(figsize=(12,5))
plt.bar(year.index.astype(str), year.values)

plt.title("Distribution of Houses by Year Built")
plt.xlabel("Year Built")
plt.ylabel("Number of Houses")

plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig("Figure4_3_YearBuilt.png", dpi=300)
plt.show()