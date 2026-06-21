"""
Step 1: Clean raw IPO data and engineer features for listing-gain prediction.
"""
import pandas as pd
import numpy as np

df = pd.read_excel('/home/claude/ipo_project/data/ipo_raw.xlsx')

# Drop rows with missing target or core subscription data — can't train on these
df = df.dropna(subset=['Listing Gain', 'QIB', 'HNI', 'RII', 'Total']).copy()

# --- Feature engineering ---
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Issue size is heavily right-skewed (23 cr to 27,858 cr) -> log transform
df['Log_Issue_Size'] = np.log1p(df['Issue_Size(crores)'])

# Retail vs institutional appetite ratio — a classic IPO analyst signal
df['RII_to_QIB_Ratio'] = df['RII'] / df['QIB'].replace(0, np.nan)
df['RII_to_QIB_Ratio'] = df['RII_to_QIB_Ratio'].fillna(0)

# Was this IPO oversubscribed overall? (Total > 1x means demand exceeded supply)
df['Is_Oversubscribed'] = (df['Total'] > 1).astype(int)

# How "hot" was institutional demand specifically (QIBs are the smart money)
df['QIB_Dominance'] = df['QIB'] / df['Total'].replace(0, np.nan)
df['QIB_Dominance'] = df['QIB_Dominance'].fillna(0)

# Binary classification target: did it list at a premium?
df['Listed_At_Premium'] = (df['Listing Gain'] > 0).astype(int)

feature_cols = [
    'Log_Issue_Size', 'QIB', 'HNI', 'RII', 'Total',
    'RII_to_QIB_Ratio', 'Is_Oversubscribed', 'QIB_Dominance',
    'Offer Price', 'Year'
]

model_df = df[feature_cols + ['Listing Gain', 'Listed_At_Premium', 'IPO_Name', 'Date']].copy()
model_df = model_df.dropna()

print(f"Final modeling dataset: {model_df.shape[0]} rows, {len(feature_cols)} features")
print(f"\nFeature columns: {feature_cols}")
print(f"\nClass balance (Listed_At_Premium): \n{model_df['Listed_At_Premium'].value_counts(normalize=True)}")

model_df.to_csv('/home/claude/ipo_project/data/ipo_clean.csv', index=False)
print("\nSaved to data/ipo_clean.csv")
