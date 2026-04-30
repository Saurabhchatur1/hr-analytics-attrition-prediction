import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self):
        self.columns = None

    def clean_data(self, df: pd.DataFrame):
        df = df.copy()

        # Drop duplicates
        df = df.drop_duplicates()

        # Handle missing values
        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

        return df

    def encode(self, df: pd.DataFrame):
        df = df.copy()

        for col in df.select_dtypes(include="object").columns:
            df[col] = pd.Categorical(df[col]).codes

        return df

    def fit_transform(self, df: pd.DataFrame):
        df = self.clean_data(df)
        df = self.encode(df)
        self.columns = df.columns
        return df

    def transform(self, df: pd.DataFrame):
        df = self.clean_data(df)
        df = self.encode(df)

        # Align columns (IMPORTANT FIX)
        df = df.reindex(columns=self.columns, fill_value=0)

        return df