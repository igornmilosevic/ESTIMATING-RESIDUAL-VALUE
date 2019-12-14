import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import LinearSVR, SVR
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
import time

start = time.time()
path = '../data/'
train = pd.read_csv(path + "Train.csv", parse_dates=['saledate'])
test = pd.read_csv(path + "Valid.csv", parse_dates=['saledate'])

print(test.head(10))
print(train.shape)
print(test.shape)
test_price =  pd.read_csv(path + "ValidSolution.csv")

test_price.drop('Usage', inplace=True, axis=1)

test = pd.merge(test, test_price, on='SalesID', how='left')

train = pd.concat([train, test], axis=0)
#
train = train.ix[train['MachineHoursCurrentMeter'].isnull()==False, :]

train = train.ix[train['fiProductClassDesc'] == 'Backhoe Loader - 14.0 to 15.0 Ft Standard Digging Depth', :].reset_index(drop=True)
print(train.shape)

for col in train.columns:
    if (len(train[col].unique())<3) | (train[col].isnull().sum(axis=0)/train.shape[0]>0.8):
        train.drop(col, axis=1, inplace=True)

train['saleyear'] = train['saledate'].dt.year
train['salemonth'] = train['saledate'].dt.month
# train['missing'] = train.ix[:, train.columns != 'SalePrice'].isnull().sum(axis=1)

eco = pd.read_csv(path + 'economic_indicators_by_month.csv')
eco['SaleDate'] = pd.to_datetime(eco['SaleDate'])
eco['saleyear'] = eco['SaleDate'].dt.year
eco['salemonth'] = eco['SaleDate'].dt.month

eco.drop(['Unnamed: 0', 'SaleDate'], inplace=True, axis=1)

train = pd.merge(train, eco, on=['salemonth', 'saleyear'], how='left')

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

train['age'] = train['saleyear'] - train['YearMade']

train['age_imputed'] = 0

train.ix[train['age']<0, 'age_imputed'] = -1

train.ix[train['age']>50, 'age_imputed'] = 1

train.drop(['saleyear'], axis=1, inplace=True)

numeric_col = ['MachineHoursCurrentMeter', 'SalePrice', 'cluster']

train = train.sort_values(by=['MachineID', 'saledate'])

train['machine_count'] = (train['MachineID'].shift(1) == train['MachineID']).astype(int)



# fill_current_meter = train.groupby('ProductGroup')['MachineHoursCurrentMeter'].mean().reset_index()
#
# fill_current_meter.columns.values[1] = 'machine_meter'
#
# train = pd.merge(train, fill_current_meter, on='ProductGroup', how='left')

# train['MachineHoursCurrentMeter_missing'] = train['MachineHoursCurrentMeter'].isnull()
train['MachineHoursCurrentMeter'].fillna(train['MachineHoursCurrentMeter'].mean(), inplace=True)



# train.drop('machine_meter', axis=1, inplace=True)

train = train.sort_values(by=['saledate', 'SalesID'])

for col in train.columns:
    # if ((col != 'MachineHoursCurrentMeter') & (col != 'SalePrice') & (col != 'cluster') & (col != 'month') & (col != 'saledate')):
    if col not in ['SalesID','missing','MachineHoursCurrentMeter_missing', 'MachineHoursCurrentMeter', 'SalePrice', 'cluster', 'month',
                   'saledate', 'age', 'age_imputed', 'machine_count', 'MachineID', 'salemonth', 'CPI', 'INDPROD', 'GDP', 'PPI']:
        print(col)
        if train[col].isnull().sum(axis=0) > 0.8 * train.shape[0]:
            train.drop(col, axis=1, inplace=True)
        else:
            # if train[col].isnull().sum(axis=0) > 0.2 * train.shape[0]:
            #     train[col + '_missing'] = train[col].isnull()
            col_groupby = train.groupby([col, 'cluster'])['SalePrice'].mean().reset_index()
            col_groupby.columns.values[2] = (col + "_price")
            col_groupby = col_groupby.sort_values(by=[col, 'cluster'])
            col_groupby['same_col'] = (col_groupby[col] == col_groupby[col].shift(1)).astype(int)
            col_groupby[col + "_price"] = col_groupby[col + "_price"].shift(1) * col_groupby['same_col']
            col_groupby.drop('same_col', axis=1, inplace=True)
            train = pd.merge(train, col_groupby, on=['cluster', col], how='left')
            if col not in ['state']:
                train.drop(col, inplace=True, axis=1)
    if col in ['state']:
        col_groupby = train.groupby([col, 'cluster'])['SalesID'].count().reset_index()
        col_groupby.columns.values[2] = (col + "_count")
        col_groupby = col_groupby.sort_values(by=[col, 'cluster'])
        col_groupby['same_col'] = (col_groupby[col] == col_groupby[col].shift(1)).astype(int)
        col_groupby[col + "_count"] = col_groupby[col + "_count"].shift(1) * col_groupby['same_col']
        col_groupby.drop('same_col', axis=1, inplace=True)
        train = pd.merge(train, col_groupby, on=['cluster', col], how='left')
        train.drop(col, inplace=True, axis=1)



train.sample(1000).to_csv('study_processing.csv')
train.fillna(-1, inplace=True)

print(train.head(10))

train = train.ix[train['saledate'] > datetime(2000,12,31), :]

# val 3
val3_condition = (train['saledate'] < datetime(2012,1,1)) & (train['saledate'] > datetime(2011,8,31))
train3_condition = (train['saledate'] < datetime(2011,9,1))

# val 2
val2_condition = (train['saledate'] < datetime(2011,9,1)) & (train['saledate'] > datetime(2011,4,30))
train2_condition = (train['saledate'] < datetime(2011,5,1))

# val 1
val1_condition = (train['saledate'] < datetime(2011,5,1)) & (train['saledate'] > datetime(2010,12,31))
train1_condition = (train['saledate'] < datetime(2011,1,1))

val_conditions = [val1_condition, val2_condition, val3_condition]
train_conditions = [train1_condition, train2_condition, train3_condition]

print(zip(val_conditions, train_conditions))

train = train.fillna(-1)
i=0

for val_index, train_index in zip(val_conditions, train_conditions):
    df, df_val = train[train_index], train[val_index]

    y, y_val = np.log(df['SalePrice']), np.log(df_val['SalePrice'])
    df.drop(['SalesID', 'SalePrice', 'saledate', 'month', 'MachineID'], axis=1, inplace=True)
    df_val.drop(['SalesID','SalePrice', 'saledate', 'month', 'MachineID'], axis=1, inplace=True)

    df_val_copy = df_val.copy()
    scale = StandardScaler()
    df = scale.fit_transform(df)
    df_val = scale.transform(df_val)
    model = MLPRegressor(random_state=12)
    model.fit(df, y)

    pred = model.predict(df_val)
    df_val_copy['pred'] = np.exp(pred)
    df_val_copy['actual'] = np.exp(y_val)
    print(np.sqrt(mean_squared_error(y_val, pred)))
    df_val_copy.to_csv('Validation_set_2_' + str(i) + '_predictions_NN.csv', index=False)

    i = i+1


train = train.sort_values(by=['saledate', 'SalesID'])

valid = train.ix[train['saledate']> datetime(2011,12,31), :].reset_index(drop=True)
train = train.ix[train['saledate']<datetime(2012,1,1), :].reset_index(drop=True)

y, y_test = np.log(train['SalePrice']), np.log(valid['SalePrice'])
valid.drop(['SalesID','SalePrice', 'saledate', 'month'], axis=1, inplace=True)
train.drop(['SalesID','SalePrice', 'saledate', 'month'], axis=1, inplace=True)


valid_copy = valid.copy()

scale = StandardScaler()
train = scale.fit_transform(train)
valid = scale.transform(valid)

model = MLPRegressor(random_state=12)
model.fit(train, y.values)

pred = model.predict(valid)

valid_copy['pred'] = np.exp(pred)
valid_copy['actual'] = np.exp(y_test)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error
print("RMSE: ", np.sqrt(mean_squared_error(valid_copy['actual'], valid_copy['pred'])))
print("MAE: ", mean_absolute_error(valid_copy['actual'], valid_copy['pred']))
print("Correlation Coeff test set: ", valid_copy['actual'].corr(valid_copy['pred']))

valid_copy.to_csv('Validation_set_2_4_predictions_NN.csv', index=False)
end = time.time()
print("Training time NN: ", end - start)