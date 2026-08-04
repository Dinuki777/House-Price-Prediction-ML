import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

avg_price = df.groupby("OverallQual")["SalePrice"].mean()

plt.figure(figsize=(8,5))
plt.bar(avg_price.index.astype(str), avg_price.values)

plt.title("Average House Price by Overall Quality")
plt.xlabel("Overall Quality")
plt.ylabel("Average Sale Price ($)")

plt.tight_layout()
plt.savefig("Figure4_4_AvgPrice.png", dpi=300)
plt.show()