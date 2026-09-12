import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
    def process_and_scale(self, df, is_training=True):
        if len(df.columns) == 1:
            col_name = df.columns[0]
            if df[col_name].astype(str).str.contains(',').any():
                df_expanded = df[col_name].astype(str).str.split(',', expand=True)
                novas_cols = str(col_name).split(',')
                if len(novas_cols) == df_expanded.shape[1]:
                    df_expanded.columns = novas_cols
                df = df_expanded
                
        df.columns = [str(col).strip().replace('"', '').replace("'", "") for col in df.columns]
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip(' "\'')
        
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(axis=1, how='all')
        
        if 'Time' in df.columns: df = df.drop(columns=['Time'])
        if 'Amount' in df.columns:
            df['Amount'] = df['Amount'].fillna(0).apply(lambda x: max(x, 0))
            df['amount_log'] = np.log1p(df['Amount'])
            df = df.drop(columns=['Amount'])
            
        if 'Class' in df.columns:
            df = df.dropna(subset=['Class']) 
        df = df.fillna(0) 
        
        y = df['Class'].values if 'Class' in df.columns else np.zeros(len(df))
        X = df.drop(columns=['Class']).values if 'Class' in df.columns else df.values
        
        X_scaled = self.scaler.fit_transform(X) if is_training else self.scaler.transform(X)
        return X_scaled, y, df