import pandas as pd

df = pd.read_csv(r"C:\Users\Dragos\OneDrive - Universitatea Politehnica Bucuresti\Desktop\DisertatiePractic\data\ndvi_bucuresti_ilfov_monthly_multi (3).csv")

# păstrăm doar coloanele utile
df = df[["date", "site", "ndvi"]]

# înlocuim placeholder-ul cu NaN
df["ndvi"] = df["ndvi"].replace(-9999, pd.NA)

# opțional: scoți lunile fără date
df = df.dropna(subset=["ndvi"])

# salvezi varianta curată
df.to_csv("ndvi_clean.csv", index=False)

print(df.head())
print(df.shape)