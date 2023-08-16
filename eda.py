import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("abalone.csv")
df["Age"] = df["Rings"] + 1.5
print(df.head())

print("number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print(df.describe().T)

'''
gender_nums = df["Sex"].value_counts()
labels = gender_nums.index
values = gender_nums.values
plt.pie(values, labels = labels)
plt.show()
'''

print(df.groupby("Sex").mean())

#plot each column except Sex and Rings relative to Age
fig, ax = plt.subplots(3, 3)
fig.tight_layout()
explanatory_vars = df.drop(["Sex", "Age", "Rings"], axis = 1)
for i, col in enumerate(explanatory_vars.columns):
    ax[i // 3, i % 3].scatter(explanatory_vars[col], df["Age"])
    ax[i // 3, i % 3].set_xlabel(col)
    ax[i // 3, i % 3].set_ylabel("Age")
    #plt.scatter(df[col], df["Age"])
    #plt.show()
plt.show()