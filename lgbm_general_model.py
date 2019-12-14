import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.svm import LinearSVR
import time

start = time.time()
path = '../data/'
train = pd.read_csv(path + "Train.csv", parse_dates=['saledate'])
test = pd.read_csv(path + "Valid.csv", parse_dates=['saledate'])

test_price =  pd.read_csv(path + "ValidSolution.csv")

test_price.drop('Usage', inplace=True, axis=1)

test = pd.merge(test, test_price, on='SalesID', how='left')

train = pd.concat([train, test], axis=0)

train = train.ix[train['MachineHoursCurrentMeter'].isnull()==False, :]

train = train.ix[train['fiProductClassDesc'] == 'Backhoe Loader - 14.0 to 15.0 Ft Standard Digging Depth', :].reset_index(drop=True)
print(train.shape)


"""
'Backhoe Loader - 14.0 to 15.0 Ft Standard Digging Depth'
'Motorgrader - 145.0 to 170.0 Horsepower'
'Skid Steer Loader - 1351.0 to 1601.0 Lb Operating Capacity'
'Track Type Tractor, Dozer - 260.0 + Horsepower'
"""

print(train.shape[0])



train = train.sort_values(by=['ModelID', 'saledate'], ascending=True)

train['saleyear'] = train['saledate'].dt.year
train['salemonth'] = train['saledate'].dt.month
train['saledow'] = train['saledate'].dt.dayofweek
train['saleday'] = train['saledate'].dt.day
# train['missing'] = train.ix[:, train.columns != 'SalePrice'].isnull().sum(axis=1)

print(train.head(5))

eco = pd.read_csv(path + 'economic_indicators_by_month.csv')
eco['SaleDate'] = pd.to_datetime(eco['SaleDate'])
eco['saleyear'] = eco['SaleDate'].dt.year
eco['salemonth'] = eco['SaleDate'].dt.month

eco.drop(['Unnamed: 0', 'SaleDate'], inplace=True, axis=1)

print(eco.columns)

train = pd.merge(train, eco, on=['salemonth', 'saleyear'], how='left')

train = train.sort_values(by='saledate', ascending=True)

train.ix[train['YearMade'] <1960, 'YearMade'] = np.NaN

train['age'] =  train['saleyear'] - train['YearMade']

state_popularity = train.groupby(['state'])['SalesID'].count().reset_index()
state_popularity.columns.values[1] = 'state_popularity'
state_popularity['state_popularity'] = state_popularity['state_popularity'].rank()

monthly_sales_state = train.groupby(['saleyear', 'salemonth', 'state'])['SalesID'].count().reset_index()
monthly_sales_state.columns.values[3] = 'state_sale_count'

model_popularity_state = train.groupby(['ModelID', 'state'])['SalesID'].count().reset_index()
model_popularity_state.columns.values[2] = 'model_popularity_count'

base_model_popularity_state = train.groupby(['fiBaseModel', 'state'])['SalesID'].count().reset_index()
base_model_popularity_state.columns.values[2] = 'base_model_popularity_count'

train = pd.merge(train, model_popularity_state, on=['ModelID', 'state'], how='left')
train = pd.merge(train, base_model_popularity_state, on=['fiBaseModel', 'state'], how='left')
train['popularity_ratio'] = train['model_popularity_count']/train['base_model_popularity_count']

expensive_state = train.groupby('state')['SalePrice'].mean().reset_index()
expensive_state['SalePrice'] = expensive_state['SalePrice'].rank()
expensive_state.columns.values[1] = 'state_exp'

train = pd.merge(train, monthly_sales_state, on=['saleyear', 'salemonth', 'state'], how='left')
train = pd.merge(train, expensive_state, on=['state'], how='left')




train['month'] = train['saledate'].dt.month + 12*(train['saleyear']-1989)

cluster = pd.DataFrame()
cluster['month'] = train['month'].unique()
cluster_number = 0

cluster = cluster.sort_values(by='month', ascending=True).reset_index(drop=True)


for i in range(0, cluster['month'].shape[0]):
    if cluster.ix[i, 'month'] % 4 == 0:
       cluster.ix[i, 'cluster'] = cluster_number
       cluster_number += 1
    else:
       cluster.ix[i, 'cluster'] = cluster_number

train = pd.merge(train, cluster, on='month', how='left')

historical_prices = train.groupby(['ModelID', 'cluster'])['SalePrice'].mean().reset_index()
historical_prices.columns.values[2] = 'historical_mean_price'

historical_prices = historical_prices.sort_values(by=['ModelID', 'cluster'])
historical_prices['same_model'] = (historical_prices['ModelID'] == historical_prices['ModelID'].shift(1)).astype(int)
historical_prices['historical_mean_price_1'] = historical_prices['historical_mean_price'].shift(1) * historical_prices['same_model']

historical_counts = train.groupby(['ModelID', 'cluster'])['SalesID'].count().reset_index()
historical_counts.columns.values[2] = 'historical_mean_count'

historical_counts = historical_counts.sort_values(by=['ModelID', 'cluster'])
historical_counts['same_model'] = (historical_counts['ModelID'] == historical_counts['ModelID'].shift(1)).astype(int)
historical_counts['historical_mean_count_1'] = historical_counts['historical_mean_count'].shift(1) * historical_counts['same_model']

historical_prices.drop(['same_model', 'historical_mean_price'], inplace=True, axis=1)
historical_counts.drop(['same_model', 'historical_mean_count'], inplace=True, axis=1)

historical_mp = train.groupby(['MachineID', 'cluster'])['SalePrice'].mean().reset_index()
historical_mp.columns.values[2] = 'historical_machine_price'

historical_mp = historical_mp.sort_values(by=['MachineID', 'cluster'])
historical_mp['same_machine'] = (historical_mp['MachineID'] == historical_mp['MachineID'].shift(1)).astype(int)
historical_mp['historical_machine_price'] = historical_mp['historical_machine_price'].shift(1) * historical_mp['same_machine']

historical_mp.drop(['same_machine'], inplace=True, axis=1)


train = pd.merge(train, historical_prices, on=['ModelID', 'cluster'], how='left')
train = pd.merge(train, historical_counts, on=['ModelID', 'cluster'], how='left')
train = pd.merge(train, historical_mp, on=['MachineID', 'cluster'], how='left')


categorical = []

for col in train.columns:
    if (len(train[col].unique())<3) | (train[col].isnull().sum(axis=0)/train.shape[0]>0.8):
        train.drop(col, axis=1, inplace=True)
    else:
        # if train[col].isnull().sum(axis=0) > 0.2 * train.shape[0]:
        #     new_col = 'missing_' + col
        #     train[new_col] = (train[col].isnull()).astype(int)
        if train[col].isnull().sum(axis=0) > 0.8 * train.shape[0]:
            train.drop(col, inplace=True, axis=1)
        elif train[col].dtype == object:
            categorical.append(col)



train['usage'] = train['MachineHoursCurrentMeter']/ (1+train['age'])

# train.sample(1000).to_csv("processed_dataset", index=False)

for col in train.columns:
    if train[col].dtype == object:
        categorical.append(col)
        train[col] = train[col].astype(str)
        label = LabelEncoder()
        train[col] = label.fit_transform(train[col])

print(train.shape)

print(train.isnull().sum(axis=0)/train.shape[0])

train = train.ix[train['saledate'] > datetime(2000,12,31), :]


train = train.sort_values(by=['saledate', 'SalesID'])

# original validation set
val3_condition = (train['saledate'] < datetime(2012,1,1)) & (train['saledate'] > datetime(2011,8,31))
train3_condition = (train['saledate'] < datetime(2011,9,1))

# val 3
val2_condition = (train['saledate'] < datetime(2011,9,1)) & (train['saledate'] > datetime(2011,4,30))
train2_condition = (train['saledate'] < datetime(2011,5,1))

# val 2
val1_condition = (train['saledate'] < datetime(2011,5,1)) & (train['saledate'] > datetime(2010,12,31))
train1_condition = (train['saledate'] < datetime(2011,1,1))

val_conditions = [val1_condition, val2_condition, val3_condition]
train_conditions = [train1_condition, train2_condition, train3_condition]

print(zip(val_conditions, train_conditions))

params = {
    'max_depth': 6,
    'num_leaves': 30,
    'objective': 'regression',
    'learning_rate': 0.05,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'metric': 'rmse',
    'num_threads': -1,
    'verbose':-1,

}
train = train.fillna(-1)


MAX_ROUNDS = 1000
val_score = 0
i=0

for val_index, train_index in zip(val_conditions, train_conditions):
    df, df_val = train[train_index], train[val_index]
    print(df_val.shape)
    print(df.shape)
    y, y_val = np.log(df['SalePrice']), np.log(df_val['SalePrice'])
    df.drop(['saledate', 'SalePrice', 'SalesID','saleyear', 'saledow', 'saleday'], axis=1, inplace=True)
    df_val.drop(['saledate', 'SalePrice', 'SalesID','saleyear', 'saledow', 'saleday'], axis=1, inplace=True)

    dtrain = lgb.Dataset(df, y, categorical_feature=categorical)
    dval = lgb.Dataset(df_val, y_val, categorical_feature=categorical)
    watchlist = [dtrain, dval]
    model = lgb.train(
               params, dtrain, num_boost_round=MAX_ROUNDS,
               valid_sets=watchlist, verbose_eval=100
            )
    pred = model.predict(df_val)
    print(np.sqrt(mean_squared_error(y_val, pred)))
    df_val['pred'] = np.exp(pred)
    df_val['actual'] = np.exp(y_val)
    df_val.to_csv('Validation_set_' + str(i) + '_predictions_lgb_PG.csv', index=False)
    i = i+1


train = train.sort_values(by=['saledate', 'SalesID'])

valid = train.ix[train['saledate']> datetime(2011,12,31), :].reset_index(drop=True)
train = train.ix[train['saledate']<datetime(2012,1,1), :].reset_index(drop=True)

train.drop(['SalesID'], axis=1, inplace=True)
valid.drop(['SalesID'], axis=1, inplace=True)

y, y_test = np.log(train['SalePrice']), np.log(valid['SalePrice'])
valid.drop(['saledate', 'SalePrice','saleyear', 'saledow', 'saleday'], axis=1, inplace=True)
train.drop(['saledate', 'SalePrice','saleyear', 'saledow', 'saleday'], axis=1, inplace=True)

dtrain = lgb.Dataset(train, y, categorical_feature=categorical)
dval = lgb.Dataset(valid, y_test, categorical_feature=categorical)
watchlist = [dtrain, dval]
model = lgb.train(
    params, dtrain, num_boost_round=MAX_ROUNDS,
    valid_sets=watchlist, verbose_eval=100
)

pred = model.predict(valid)



valid['pred'] = np.exp(pred)
valid['actual'] = np.exp(y_test)

valid.to_csv('Validation_set_4_predictions_lgb_PG.csv', index=False)
from sklearn.metrics import mean_absolute_error
print("RMSE: ", np.sqrt(mean_squared_error(valid['actual'], valid['pred'])))
print("MAE: ", mean_absolute_error(valid['actual'], valid['pred']))
print("Correlation Coeff test set: ", valid['actual'].corr(valid['pred']))

df = pd.DataFrame()
df['feat'] = train.columns
df['importtance'] = model.feature_importance('split')

df.to_csv("store_feat_importance_lgb.csv")

end = time.time()
print("Training time LGB: ", end - start)