import pandas as pd
import numpy as np
from sklearn.svm import LinearSVR, SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

val_0_RF = pd.read_csv('Validation_set_0_predictions_RF.csv')
val_0_nn = pd.read_csv('Validation_set_2_0_predictions_NN.csv')
val_0_lgb = pd.read_csv('Validation_set_0_predictions_lgb_PG.csv')

val_0_set = pd.DataFrame()
val_0_set['RF_predictions'] = val_0_RF['pred']
val_0_set['nn_predictions'] = val_0_nn['pred']
val_0_set['lgb_predictions'] = val_0_lgb['pred']
print(val_0_set.isnull().sum(axis=0))
y_val0 = val_0_RF['actual']

val_1_RF = pd.read_csv('Validation_set_1_predictions_RF.csv')
val_1_nn = pd.read_csv('Validation_set_2_1_predictions_NN.csv')
val_1_lgb = pd.read_csv('Validation_set_1_predictions_lgb_PG.csv')

val_1_set = pd.DataFrame()
val_1_set['RF_predictions'] = val_1_RF['pred']
val_1_set['nn_predictions'] = val_1_nn['pred']
val_1_set['lgb_predictions'] = val_1_lgb['pred']
print(val_1_set.isnull().sum(axis=0))
print(val_1_set.head(5))
y_val1 = val_1_RF['actual']


val_2_RF = pd.read_csv('Validation_set_2_predictions_RF.csv')
val_2_nn = pd.read_csv('Validation_set_2_2_predictions_NN.csv')
val_2_lgb = pd.read_csv('Validation_set_2_predictions_lgb_PG.csv')

val_2_set = pd.DataFrame()
val_2_set['RF_predictions'] = val_2_RF['pred']
val_2_set['nn_predictions'] = val_2_nn['pred']
val_2_set['lgb_predictions'] = val_2_lgb['pred']
print(val_2_set.isnull().sum(axis=0))
print(val_2_set.head(5))
y_val2 = val_2_RF['actual']

ind_vals = [-9, -3, 3, 9]
machine_vals = [-50, 0, 50]

train = pd.concat([val_0_set, val_1_set, val_2_set], axis=0)
y_train = pd.concat([y_val0, y_val1, y_val2], axis=0)

scale = StandardScaler()
train = scale.fit_transform(train)

start = time.time()
model = LinearSVR(C=4000, random_state=10)

model.fit(train, y_train.values)
end = time.time()


predictions = pd.DataFrame()
counter = 1
for ind_val in ind_vals:
    for machine_val in machine_vals:
        val_4_RF = pd.read_csv("Validation_set_4_predictions_RF_" + str(ind_val) + "_" + str(machine_val)  + "_.csv")
        val_4_nn = pd.read_csv("Validation_set_2_4_predictions_NN_" + str(ind_val) + "_" + str(machine_val) + "_.csv")
        val_4_lgb = pd.read_csv("Validation_set_4_predictions_lgb_PG_" + str(ind_val) + "_" + str(machine_val) + "_.csv")

        val_4_set = pd.DataFrame()
        val_4_set['RF_predictions'] = val_4_RF['pred']
        val_4_set['nn_predictions'] = val_4_nn['pred']
        val_4_set['lgb_predictions'] = val_4_lgb['pred']

        y_val4 = val_4_RF['actual']

        val_4_set = scale.transform(val_4_set)


        pred = model.predict(val_4_set)

        pred_ = 'pred_' +  str(ind_val) + "_" + str(machine_val)
        predictions['valid'] = y_val4
        predictions[pred_] = pred

        from sklearn.metrics import mean_absolute_error

        print("Iteration identifier " + str(ind_val) + " & " + str(machine_val))
        print("logRMSE: ",  np.sqrt(mean_squared_error(np.log(y_val4), np.log(pred))))
        print("RMSE: ", np.sqrt(mean_squared_error(y_val4, pred)))
        print("MAE: ", mean_absolute_error(y_val4, pred))
        print("Correlation Coeff test set: ", predictions['valid'].corr(predictions[pred_]))
        counter +=1

# predictions.to_csv('predictions.csv', index=False)
# y_val4 = predictions['valid']
# predictions.drop('valid', axis=1, inplace=True)
# predictions['avg_sales_price'] = predictions.mean(axis=1)
# pred = predictions['avg_sales_price']
# predictions['valid'] = y_val4
# print("logRMSE: ", np.sqrt(mean_squared_error(np.log(y_val4), np.log(pred))))
# print("RMSE: ", np.sqrt(mean_squared_error(y_val4, pred)))
# print("MAE: ", mean_absolute_error(y_val4, pred))
# print("Correlation Coeff test set: ", predictions['valid'].corr(predictions['avg_sales_price']))
