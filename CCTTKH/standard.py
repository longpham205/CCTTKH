import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, LabelEncoder

df = pd.read_csv("glass.csv")

print("Dữ liệu gốc:")
print(df.head())

if "Type" in df.columns:
    encoder = LabelEncoder()
    df["Type_encoded"] = encoder.fit_transform(df["Type"])

scaler1 = MinMaxScaler()
df["col0_scaled"] = scaler1.fit_transform(df.iloc[:, [0]])

cols2 = df.columns[1:3]
scaler2 = StandardScaler()
df[cols2] = scaler2.fit_transform(df[:, 1:2])

cols3 = df.columns[3:6]
scaler3 = Normalizer()
df[cols3] = scaler3.fit_transform(df[cols3])

print("\nDữ liệu sau chuẩn hoá:")
print(df.head())
