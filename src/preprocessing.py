import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats

def handle_missing_and_clean(df):
    # Eksikleri medyan ile doldur (Outlier'a karşı dayanıklI olması için)
    df['MMSE'] = df['MMSE'].fillna(df['MMSE'].median())
    
    # Z-Score Temizliği: Verideki 'çelişkili' gürültülü satırları atıyoruz
    num_cols = ['Age', 'EDUC', 'MMSE', 'eTIV', 'nWBV']
    z_scores = np.abs(stats.zscore(df[num_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    return df

def engineer_features(df):
    # En kritik oran: Beyin hacminin kafa hacmine oranı (Atrofi) 
    df['Atrophy_Rate'] = df['nWBV'] / df['eTIV']
    return df 

def get_preprocessor():
    # Sadece en 'temiz' ve 'belirleyici' sütunları bıraktık
    numeric_features = ['Age', 'EDUC', 'MMSE', 'eTIV', 'nWBV', 'Atrophy_Rate']
    categorical_features = ['M/F']

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop='first'))])

    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])