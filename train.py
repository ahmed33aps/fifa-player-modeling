"""Leakage-aware baseline for the supplied FIFA dataset, alongside original notebooks."""
from pathlib import Path
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, accuracy_score

def run():
    data=pd.read_csv(Path(__file__).with_name('Fifa.csv')).drop_duplicates()
    numeric=['Age','Overall_Rating','Future Potential','Value Per M$','Total_Stats Score']
    for col in numeric:data[col]=pd.to_numeric(data[col],errors='coerce')
    data=data.dropna(subset=['Value Per M$','Overall_Rating'])
    data=data.loc[data['Value Per M$']>0]
    train,test=train_test_split(data,test_size=.2,random_state=42)
    # Tiers are defined by the rating, so rating itself must not be a classifier input.
    cutoffs=train['Overall_Rating'].quantile([.25,.5,.75]).values
    def tier(frame):
        import numpy as np
        return np.searchsorted(cutoffs,frame['Overall_Rating'].values,side='left')
    def pipeline(features,model):
        nums=[x for x in features if x in numeric]
        cats=[x for x in features if x not in numeric]
        prep=ColumnTransformer([('num',SimpleImputer(strategy='median'),nums),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),cats)])
        return Pipeline([('prepare',prep),('model',model)])
    reg_features=['Age','Overall_Rating','Future Potential','Total_Stats Score','Country','Position']
    cls_features=['Age','Future Potential','Country','Position']
    reg=pipeline(reg_features,RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=2,min_samples_leaf=3))
    cls=pipeline(cls_features,RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=2,min_samples_leaf=3))
    reg.fit(train[reg_features],train['Value Per M$']);cls.fit(train[cls_features],tier(train))
    pred=reg.predict(test[reg_features])
    result={'rows':len(data),'split':'random 80/20 seed 42','mae_value_millions':mean_absolute_error(test['Value Per M$'],pred),'rmse_value_millions':root_mean_squared_error(test['Value Per M$'],pred),'r2':r2_score(test['Value Per M$'],pred),'tier_accuracy':accuracy_score(tier(test),cls.predict(test[cls_features])),'train_only_rating_cutoffs':cutoffs.tolist(),'caution':'Tier excludes its defining rating and total stats from classifier inputs. No future-season generalization claim.'}
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':run()
