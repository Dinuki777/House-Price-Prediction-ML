import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

garage = df["GarageCars"].value_counts().sort_index()

plt.figure(figsize=(8,5))
plt.bar(garage.index.astype(str), garage.values)

plt.title("Distribution of Garage Capacity")
plt.xlabel("Garage Cars")
plt.ylabel("Number of Houses")

plt.tight_layout()
plt.savefig("Figure4_2_GarageCars.png", dpi=300)
plt.show()