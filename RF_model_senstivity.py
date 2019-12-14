import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load


path = '../data/'
train = pd.read_csv(path + "Train.csv", parse_dates=['saledate'])
test = pd.read_csv(path + "Valid.csv", parse_dates=['saledate'])

test_price =  pd.read_csv(path + "ValidSolution.csv")

test_price.drop('Usage', inplace=True, axis=1)

test = pd.merge(test, test_price, on='SalesID', how='left')

train = pd.concat([train, test], axis=0)

# # ModeID=4605
# # ModeID=3362
# # ModeID=9551
# # ModeID=1169
# # ModeID=4147
# ModeID=4579
#
# train = train.ix[train['ModelID']== ModeID, :]

print(train.shape)
train = train.ix[train['MachineHoursCurrentMeter'].isnull()==False, :]




train = train.sort_values(by=['ModelID', 'saledate'], ascending=True)

train['saleyear'] = train['saledate'].dt.year
train['salemonth'] = train['saledate'].dt.month
train['saledow'] = train['saledate'].dt.dayofweek
train['saleday'] = train['saledate'].dt.day
# train['missing'] = train.ix[:, train.columns != 'SalePrice'].isnull().sum(axis=1)

eco = pd.read_csv(path + 'economic_indicators_by_month.csv')
eco['SaleDate'] = pd.to_datetime(eco['SaleDate'])
eco['saleyear'] = eco['SaleDate'].dt.year
eco['salemonth'] = eco['SaleDate'].dt.month

eco.drop(['Unnamed: 0', 'SaleDate'], inplace=True, axis=1)

print(eco.columns)
print(train['saleyear'].value_counts())
train = pd.merge(train, eco, on=['salemonth', 'saleyear'], how='left')

print(train.head(5))


train = train.sort_values(by='saledate', ascending=True)

train.ix[train['YearMade'] <1960, 'YearMade'] = np.NaN

train['age'] =  train['saleyear'] - train['YearMade']

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

historical_prices.to_csv('check_historical_price.csv')
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

train = train.fillna(0)
i=0



for val_index, train_index in zip(val_conditions, train_conditions):
    df, df_val = train[train_index], train[val_index]
    print(df_val.shape)
    print(df.shape)
    y, y_val = np.log(df['SalePrice']), np.log(df_val['SalePrice'])
    df.drop(['saledate', 'SalePrice', 'SalesID','saleyear', 'saledow', 'saleday'], axis=1, inplace=True)
    df_val.drop(['saledate', 'SalePrice', 'SalesID','saleyear', 'saledow', 'saleday'], axis=1, inplace=True)
    df_val_copy = df_val.copy()
    model = RandomForestRegressor(random_state=2018, n_jobs=-1, max_features=0.2, min_samples_leaf=1, n_estimators=100)
    model.fit(df, y)
    pred = model.predict(df_val)
    df_val_copy['pred'] = np.exp(pred)
    df_val_copy['actual'] = np.exp(y_val)
    print(np.sqrt(mean_squared_error(y_val, pred)))
    df_val_copy.to_csv('Validation_set_' + str(i) + '_predictions_RF.csv', index=False)

    i = i+1


train = train.sort_values(by=['saledate', 'SalesID'])

valid = train.ix[train['saledate']> datetime(2011,12,31), :].reset_index(drop=True)
train = train.ix[train['saledate']<datetime(2012,1,1), :].reset_index(drop=True)

train.drop(['SalesID'], axis=1, inplace=True)
valid.drop(['SalesID'], axis=1, inplace=True)

y, y_test = np.log(train['SalePrice']), np.log(valid['SalePrice'])
valid.drop(['saledate', 'SalePrice','saleyear', 'saledow', 'saleday'], axis=1, inplace=True)
train.drop(['saledate', 'SalePrice','saleyear', 'saledow', 'saleday'], axis=1, inplace=True)


valid_copy = valid.copy()

model = RandomForestRegressor(random_state=2018, n_jobs=-1, max_features=0.2, min_samples_leaf=1, n_estimators=100)
model.fit(train, y.values)

ind_vals = [-9, -3, 3, 9]
machine_vals = [-50, 0, 50]

for ind_val in ind_vals:
    for machine_val in machine_vals:
        valid_copy = valid.copy()
        valid_copy['INDPROD'] = (1 + ind_val/100) * valid['INDPROD']
        valid_copy['MachineHoursCurrentMeter'] = (1 + machine_val/100) * valid['MachineHoursCurrentMeter']

        feat_imp = pd.DataFrame()
        feat_imp['feat']  = train.columns
        feat_imp['importance'] = model.feature_importances_

        feat_imp.to_csv('store_feat_imp.csv', index=False)

        pred = model.predict(valid_copy)

        valid_copy['pred'] = np.exp(pred)
        valid_copy['actual'] = np.exp(y_test)
        print("Iteration identifier " + str(ind_val) + " & " + str(machine_val))
        print("logRMSE: ",  np.sqrt(mean_squared_error(y_test, pred)))
        print("RMSE: ", np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred))))
        print("MAE: ", mean_absolute_error(np.exp(y_test), np.exp(pred)))
        print("Correlation Coeff test set: ", valid_copy['actual'].corr(valid_copy['pred']))

        valid_copy.to_csv("Validation_set_4_predictions_RF_" + str(ind_val) + "_" + str(machine_val)  + "_.csv", index=False)

