import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer


class handle_script():
    def __init__(self):
        self.pt = None
        self.labels = None
        self.scalers = None


    def handle_data(self,df):
        df['Value Per M$'] =df['Value Per M$'].drop(index=df[df['Value Per M$'] == 0].index).copy()
        df = df.dropna(subset='Value Per M$').copy()

        return df


    def fit_boxcox(self , df):
        self.pt = PowerTransformer(method='box-cox')
        self.pt.fit(df[['Value Per M$']])

        return self


    def transform_boxcox(self,df):
        if self.pt is None:
            raise Exception('box_cox didn\'t fit yet')
        
        df['Value Per M$_after_boxcox'] = self.pt.transform(df[['Value Per M$']]).copy()

        return df
    

    def fit_encode(self , df):
        self.labels = {}

        for col in df.select_dtypes('object').drop(columns = 'Name', errors = 'ignore').columns:
            label = LabelEncoder()
            label.fit(df[col])
            
            self.labels[col] = label

        return self


    def transform_encode(self , df):
        if self.labels is None:
            raise Exception('label encodeing didn\'t fit yet')
        
        for col in df.select_dtypes('object').drop(columns = 'Name' , errors = 'ignore').columns:
            label = self.labels[col]
            df[col+'_label'] = label.transform(df[col])

        return df
    

    def fit_scaler(self , df):
        self.scalers = {}

        for col in df.select_dtypes('number').columns:
            scaler = StandardScaler()
            scaler.fit(df[[col]])

            self.scalers[col] = scaler

        return self
    

    def transform_scaler(self , df):
        if self.scalers is None:
            raise Exception('scaler didn\'t fit yet')

        for col, scaler in self.scalers.items():
            df[col+'_scaled'] = scaler.transform(df[[col]])

        return df
    
    
    def selcet_features(self,df):

        df = df.drop(columns = ['Name', 'Country', 'Position', 'Age', 'Overall_Rating',
                                'Future Potential', 'Team', 'Value Per M$',
                                'Total_Stats Score','Value Per M$_after_boxcox',
                                'Value Per M$_scaled'])

        return df
    