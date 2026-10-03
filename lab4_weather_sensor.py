import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore

# Weather sensor data
data = {
    "Hour": range(1, 16),
    "Temperature": [
        22, 23, np.nan, np.nan, np.nan,
        25, 26, 58, 24, 23,
        22, 21, 23, 24, 25
    ]
}

df = pd.DataFrame(data)

# --------------------------------------------------
# 1. Statistics BEFORE cleaning
# --------------------------------------------------

raw_temperature = df["Temperature"].dropna()

print("Stats BEFORE cleaning:")
print("Mean:", raw_temperature.mean())
print("Std:", raw_temperature.std())
print("Skewness:", stats.skew(raw_temperature))
print("Kurtosis:", stats.kurtosis(raw_temperature))

# --------------------------------------------------
# 2. Fill missing values using linear interpolation
# --------------------------------------------------

df["Temperature_Clean"] = df["Temperature"].interpolate(
    method="linear"
)

print("\nAfter interpolation, missing values:",
      df["Temperature_Clean"].isnull().sum())

# --------------------------------------------------
# 3. Detect outlier using Z-score threshold = 2
# --------------------------------------------------

z_scores = zscore(df["Temperature_Clean"])

outlier_mask = abs(z_scores) > 2

for index in df.index[outlier_mask]:
    print(
        "Z-score(>2) outlier detected at Hour:",
        df.loc[index, "Hour"],
        "(value:", df.loc[index, "Temperature_Clean"], ")"
    )

# --------------------------------------------------
# 4. Replace outlier with median
# --------------------------------------------------

median_temperature = df["Temperature_Clean"].median()

df.loc[outlier_mask, "Temperature_Clean"] = median_temperature

print("Replaced with column median:", median_temperature)

# --------------------------------------------------
# 5. Final Z-score standardization
# --------------------------------------------------

df["Temperature_Zscore"] = zscore(
    df["Temperature_Clean"]
)

# --------------------------------------------------
# 6. Before/After comparison
# --------------------------------------------------

comparison = df[
    ["Hour", "Temperature", "Temperature_Clean", "Temperature_Zscore"]
]

print("\nBefore/After comparison:")
print(comparison)

# --------------------------------------------------
# 7. Statistics AFTER cleaning
# --------------------------------------------------

print("\nStats AFTER cleaning:")
print("Mean:", df["Temperature_Clean"].mean())
print("Std:", df["Temperature_Clean"].std())