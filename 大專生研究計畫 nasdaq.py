#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df1 = pd.read_csv("/Users/chenkaijung/Documents/大專生研究計畫/TW.csv")
df1.head()


# In[3]:


df1.drop(['Close', 'High', 'Low', 'Close', 'Volume','Adj Close'], axis=1, inplace=True)
df1['Date'] = pd.to_datetime(df1['Date'])
df1 = df1.set_index(['Date'], drop=True)
df1.tail(10)


# In[4]:


df2 = pd.read_csv("/Users/chenkaijung/Documents/大專生研究計畫/NASDAQ.csv")
df2.head()


# In[5]:


df2.drop(['Close', 'High', 'Low', 'Close', 'Volume','Adj Close'], axis=1, inplace=True)
df2['Date'] = pd.to_datetime(df2['Date'])
df2 = df2.set_index(['Date'], drop=True)
df2.head(10)


# # 台股 Linear model

# In[5]:


#資料歸一化
scaler = MinMaxScaler(feature_range=(-1, 1))

#需將資料做reshape的動作，使其shape為(資料長度,1) 
x_set= df1.values.reshape(-1,1)
TW = scaler.fit_transform(x_set)
TW_df = pd.DataFrame(TW)


# In[6]:


split_point = int(len(TW_df)*0.9)
train = TW_df.iloc[:split_point].copy()
test = TW_df.iloc[split_point:-1].copy() #因為預測一天後的結果，最後一天預測並無解答，所以-1


# In[7]:


predict_days = 1
X_train = train[:-predict_days]
y_train = train[predict_days:]
X_test = test[:-predict_days]
y_test = test[predict_days:]


# In[8]:


X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values


# In[61]:


X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[62]:


from sklearn.linear_model import LinearRegression
Linear_model = LinearRegression()
Linear_model.fit(X_train,y_train)


# In[63]:


Linear_preds = Linear_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.plot(y_test[:,0], label='TW_Open')
plt.plot(Linear_preds, label='Linear')
plt.title("Linear's Prediction")
plt.xlabel('Observation')
plt.ylabel('Open')
plt.legend()
plt.show();


# In[64]:


reduction_Linear_pred = scaler.inverse_transform(Linear_preds)
reduction_test = df1.iloc[split_point:-1].copy()
reduction_y_test = reduction_test[predict_days:]


# In[65]:


print("Model Coefficients:",  Linear_model.coef_)
print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Linear_pred ))
print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Linear_pred ))
print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Linear_pred )))
print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Linear_pred ))


# In[66]:


linear_result=pd.DataFrame(y_test)
linear_result['Predict']=Linear_preds
linear_result.columns = ['Actual_Price','Predicted_Price']


# In[68]:


linear_result


# In[69]:


linear_result.describe()


# # 台股LSTM 

# In[6]:


#資料歸一化
scaler = MinMaxScaler(feature_range=(-1, 1))

#需將資料做reshape的動作，使其shape為(資料長度,1) 
x_set= df1.values.reshape(-1,1)
TW = scaler.fit_transform(x_set)
TW_df = pd.DataFrame(TW)


# In[9]:


split_point = int(len(TW_df)*0.9)
train = TW_df.iloc[:split_point].copy()
test = TW_df.iloc[split_point:-1].copy() #因為預測一天後的結果，最後一天預測並無解答，所以-1


# In[10]:


predict_days = 1
X_train = train[:-predict_days]
y_train = train[predict_days:]
X_test = test[:-predict_days]
y_test = test[predict_days:]


# In[11]:


X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values


# In[12]:


X_train = X_train.reshape((X_train.shape[0],1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0],1, X_test.shape[1]))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[37]:


lstm_model = Sequential()
lstm_model.add(LSTM(units=400, return_sequences=False, input_shape=(X_train.shape[1],1)))
lstm_model.add(Dropout(0.2))
#lstm_model.add(LSTM(units=50))

lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])


# In[38]:


y_pred_test_lstm = lstm_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.plot(y_test[:,0], label='TW_Open')
plt.plot(y_pred_test_lstm, label='LSTM')
plt.title("LSTM's Prediction")
plt.xlabel('Observation')
plt.ylabel('Open')
plt.legend()
plt.show();


# In[39]:


reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
reduction_test = df1.iloc[split_point:-1].copy()
reduction_y_test = reduction_test[predict_days:]


# In[40]:


print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[49]:


LSTM_result=pd.DataFrame(reduction_y_test)
LSTM_result['Predict']=reduction_Lstm_pred
LSTM_result.columns = ['Actual_Price','Predicted_Price']


# ### LSTM_result

# In[50]:


LSTM_result


# In[32]:


LSTM_result.describe()


# # 台股ARIMA

# In[7]:


import warnings

warnings.filterwarnings("ignore")


# In[12]:


df = pd.read_csv("/Users/chenkaijung/Documents/大專生研究計畫/TW.csv")
df.drop(['Close', 'High', 'Low', 'Close', 'Volume','Adj Close'], axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index(['Date'], drop=True)
df.head(10)
# 將數據拆分成訓練/測試集
train = df["Open"][:int(0.9*df.shape[0])].copy()
test = df["Open"][int(0.9*df.shape[0]):].copy()


# In[22]:


from pmdarima.arima import auto_arima
model_autoARIMA = auto_arima(train, start_p=0, start_q=0,
                      test='adf',       # use adftest to find             optimal 'd'
                      max_p=4, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
print(model_autoARIMA.summary())


# In[15]:


history = [x for x in train]
model_predictions = []
N_test_observations = len(test)
for time_point in range(N_test_observations):
    model = ARIMA(history, order=(2,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = test[time_point]
    history.append(true_test_value)


# In[16]:


print("Mean Absolute Error:", mean_absolute_error(test, model_predictions))
print('Mean Squared Error:', mean_squared_error(test, model_predictions))
print('Root Mean Squared Error:', sqrt(mean_squared_error(test, model_predictions)))
print("Coefficient of Determination:", r2_score(test, model_predictions))


# In[24]:


import matplotlib.pyplot as plt
test_set_range = df[int(len(df)*0.9):].index

plt.plot(test_set_range, test, label='price')
plt.plot(test_set_range, model_predictions,label='Predicted Price')
plt.title('Open price') 
plt.xlabel('Date') 
plt.ylabel('Prices') 
plt.legend() 
plt.show()


# In[23]:


model_fit.summary()


# In[19]:


Arima_result=pd.DataFrame(test)
a=np.array(model_predictions)
Arima_result['Predict']= a
Arima_result.columns = ['Actual_Price','Predicted_Price']


# In[20]:


Arima_result


# In[21]:


Arima_result.describe()


# # 台股、美股雙變量LSTM

# In[245]:


df1 = pd.read_csv("/Users/chenkaijung/Documents/大專生研究計畫/TW.csv")
df1.drop(['Close', 'High', 'Low', 'Close', 'Volume','Adj Close'], axis=1, inplace=True)
df1['Date'] = pd.to_datetime(df1['Date'])
df1 = df1.set_index(['Date'], drop=True)
df1.head(10)


# In[246]:


df2 = pd.read_csv("/Users/chenkaijung/Documents/大專生研究計畫/NASDAQ.csv")
df2.drop(['Close', 'High', 'Low', 'Close', 'Volume','Adj Close'], axis=1, inplace=True)
df2['Date'] = pd.to_datetime(df2['Date'])
df2 = df2.set_index(['Date'], drop=True)
df2.head(10)


# In[247]:


#資料歸一化
scaler = MinMaxScaler(feature_range=(-1, 1))

#需將資料做reshape的動作，使其shape為(資料長度,1) 
y_set= df2.values.reshape(-1,1)
Nasdaq = scaler.fit_transform(y_set)
Nasdaq_df = pd.DataFrame(Nasdaq)
x_set= df1.values.reshape(-1,1)
TW = scaler.fit_transform(x_set)
TW_df = pd.DataFrame(TW)


# In[248]:


#合併兩個資料
result_df=pd.concat([TW_df , Nasdaq_df], axis = 1)
#去除nan值，暫用此方法，之後需重新整理資料
result_df = result_df.dropna()


# In[249]:


split_point = int(len(result_df)*0.9)
train = result_df.iloc[:split_point].copy()
test = result_df.iloc[split_point:-1].copy() #因為預測一天後的結果，最後一天預測並無解答，所以-1


# In[250]:


predict_days = 1
X_train = train[:-predict_days]
y_train = train[predict_days:]
X_test = test[:-predict_days]
y_test = test[predict_days:]


# In[251]:


X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values


# In[252]:


X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[294]:


lstm_model_x2 = Sequential()
lstm_model_x2.add(LSTM(units=400, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model_x2.add(Dropout(0.1))

lstm_model_x2.add(Dense(2))
lstm_model_x2.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history_lstm_model = lstm_model_x2.fit(X_train, y_train, epochs=30, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])


# In[295]:


lstm_x2 = lstm_model_x2.predict(X_test)
plt.figure(figsize=(10, 6))
plt.plot(y_test[:,0], label='TW_Open')
plt.plot(lstm_x2[:,0], label='LSTM')
plt.title("LSTM's Prediction")
plt.xlabel('Observation')
plt.ylabel('Open')
plt.legend()
plt.show();


# In[296]:


lstm_x2_reshape=lstm_x2[:,0].reshape(-1,1)
reduction_Lstm_x2_pred = scaler.inverse_transform(lstm_x2_reshape)
reduction_test = df1.iloc[split_point:-1].copy()
reduction_y_test = reduction_test[predict_days:]


# In[297]:


#print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test[:,0],lstm_x2[:,0][:,0])))
print("Mean Absolute Error:", mean_absolute_error(reduction_y_test,reduction_Lstm_x2_pred))
print('Mean Squared Error:', mean_squared_error(reduction_y_test,reduction_Lstm_x2_pred))
print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test,reduction_Lstm_x2_pred)))
print("Coefficient of Determination:", r2_score(reduction_y_test,reduction_Lstm_x2_pred))


# In[298]:


lstm_x2_result=pd.DataFrame(reduction_y_test)
lstm_x2_result['Predict']=reduction_Lstm_x2_pred
lstm_x2_result.columns = ['Actual_Price','Predicted_Price']


# In[299]:


lstm_x2_result


# In[300]:


lstm_x2_result.describe()


# # 台股、美股多變量LSTM(12項)

# In[1]:


df1 = pd.read_csv("/Users/chenkaijung/Documents/大專生研究計畫/TW.csv")
#df1.drop(['Close', 'High', 'Low', 'Close', 'Volume','Adj Close'], axis=1, inplace=True)
df1['Date'] = pd.to_datetime(df1['Date'])
df1 = df1.set_index(['Date'], drop=True)
df1 


# In[2]:


df2 = pd.read_csv("/Users/chenkaijung/Documents/大專生研究計畫/NASDAQ.csv")
df2['Date'] = pd.to_datetime(df2['Date'])
df2 = df2.set_index(['Date'], drop=True)
df2


# In[3]:


#資料歸一化
scaler = MinMaxScaler(feature_range=(-1, 1))

#需將資料做reshape的動作，使其shape為(資料長度,1) 

y_set= df2.values.reshape(-1,6)
Nasdaq = scaler.fit_transform(y_set)
Nasdaq_df = pd.DataFrame(Nasdaq)
x_set= df1.values.reshape(-1,6)
TW = scaler.fit_transform(x_set)
TW_df = pd.DataFrame(TW)


# In[4]:


#合併兩個資料
result_df=pd.concat([TW_df , Nasdaq_df], axis = 1)
#去除nan值，暫用此方法，之後需重新整理資料
result_df = result_df.dropna()
result_df


# In[59]:


split_point = int(len(result_df)*0.9)
train = result_df.iloc[:split_point].copy()
test = result_df.iloc[split_point:-1].copy() #因為預測一天後的結果，最後一天預測並無解答，所以-1


# In[60]:


predict_days = 1
X_train = train[:-predict_days]
y_train = train[predict_days:]
X_test = test[:-predict_days]
y_test = test[predict_days:]


# In[61]:


X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values


# In[62]:


X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[63]:


lstm_model = Sequential()
#lstm_model.add(LSTM(7, input_shape=(X_train.shape[1], X_train.shape[2]), activation='sigmoid', kernel_initializer='lecun_uniform'))
#lstm_model.add(Dense(1))
lstm_model.add(LSTM(units=400, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), activation='sigmoid', kernel_initializer='lecun_uniform'))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=400, return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=400, return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=400, return_sequences=False))
lstm_model.add(Dense(12))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history_lstm_model = lstm_model.fit(X_train, y_train, epochs=30, batch_size=2, verbose=1, shuffle=True, callbacks=[early_stop])


# In[64]:


lstm_x12 = lstm_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.plot(y_test[:,0], label='TW_Open')
plt.plot(lstm_x12[:,0], label='LSTM')
plt.title("LSTM's Prediction")
plt.xlabel('Observation')
plt.ylabel('Open')
plt.legend()
plt.show();


# In[65]:


lstm_x12_reshape=lstm_x12[:,0:6].reshape(-1,6)
reduction_Lstm_x12_pred = scaler.inverse_transform(lstm_x12_reshape)
reduction_Lstm_x12_pred =pd.DataFrame(reduction_Lstm_x12_pred)
reduction_test = df1.iloc[split_point:-1].copy()
reduction_y_test = reduction_test[predict_days:]


# In[66]:


#歸一化前
#print("Mean Absolute Error:", mean_absolute_error(y_test[:,0], lstm_x12[:,0]))
#print('Mean Squared Error:', mean_squared_error(y_test[:,0], lstm_x12[:,0]))
#print('Root Mean Squared Error:', sqrt(mean_squared_error(y_test[:,0], lstm_x12[:,0])))
#print("Coefficient of Determination:", r2_score(y_test[:,0], lstm_x12[:,0]))


# In[67]:


print("Mean Absolute Error:", mean_absolute_error(reduction_y_test.iloc[:,0],reduction_Lstm_x12_pred[0]))
print('Mean Squared Error:', mean_squared_error(reduction_y_test.iloc[:,0],reduction_Lstm_x12_pred[0]))
print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test.iloc[:,0],reduction_Lstm_x12_pred[0])))
print("Coefficient of Determination:", r2_score(reduction_y_test.iloc[:,0],reduction_Lstm_x12_pred[0]))


# # k-Nearest Neighbours

# In[68]:


df1 = pd.read_csv("/Users/chenkaijung/Documents/大專生研究計畫/TW.csv")
df1.drop(['Close', 'High', 'Low', 'Close', 'Volume','Adj Close'], axis=1, inplace=True)
df1['Date'] = pd.to_datetime(df1['Date'])
df1 = df1.set_index(['Date'], drop=True)
df1 


# In[69]:


#importing libraries
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV


# In[70]:


#資料歸一化
scaler = MinMaxScaler(feature_range=(-1, 1))

#需將資料做reshape的動作，使其shape為(資料長度,1) 
x_set= df1.values.reshape(-1,1)
TW = scaler.fit_transform(x_set)
TW_df = pd.DataFrame(TW)


# In[71]:


split_point = int(len(TW_df)*0.9)
train = TW_df.iloc[:split_point].copy()
test = TW_df.iloc[split_point:-1].copy() #因為預測一天後的結果，最後一天預測並無解答，所以-1


# In[72]:


predict_days = 1
X_train = train[:-predict_days]
y_train = train[predict_days:]
X_test = test[:-predict_days]
y_test = test[predict_days:]


# In[73]:


X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values


# In[74]:


X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[75]:


params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

#fit the model and make predictions
model.fit(X_train,y_train)
k_Nearest_preds = model.predict(X_test)


# In[76]:


print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test[:,0], k_Nearest_preds)))


# In[77]:


plt.figure(figsize=(10, 6))
plt.plot(y_test[:,0], label='TW_Open')
plt.plot(k_Nearest_preds , label='LSTM')
plt.title("LSTM's Prediction")
plt.xlabel('Observation')
plt.ylabel('Open')
plt.legend()
plt.show();


# # 台股ANN

# In[305]:


df1 = pd.read_csv("/Users/chenkaijung/Documents/大專生研究計畫/TW.csv")
df1.drop(['Close', 'High', 'Low', 'Close', 'Volume','Adj Close'], axis=1, inplace=True)
df1['Date'] = pd.to_datetime(df1['Date'])
df1 = df1.set_index(['Date'], drop=True)
df1 


# In[306]:


#資料歸一化
scaler = MinMaxScaler(feature_range=(-1, 1))

#需將資料做reshape的動作，使其shape為(資料長度,1) 
x_set= df1.values.reshape(-1,1)
TW = scaler.fit_transform(x_set)
TW_df = pd.DataFrame(TW)


# In[307]:


split_point = int(len(TW_df)*0.9)
train = TW_df.iloc[:split_point].copy()
test = TW_df.iloc[split_point:-1].copy() #因為預測一天後的結果，最後一天預測並無解答，所以-1


# In[308]:


predict_days = 1
X_train = train[:-predict_days]
y_train = train[predict_days:]
X_test = test[:-predict_days]
y_test = test[predict_days:]


# In[309]:


X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values


# In[310]:


X_train = X_train.reshape((X_train.shape[0],1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0],1, X_test.shape[1]))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[311]:


nn_model = Sequential()
nn_model.add(Dense(400, input_dim=1, activation='relu'))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(1))
nn_model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history = nn_model.fit(X_train, y_train, epochs=30, batch_size=1, verbose=1, callbacks=[early_stop], shuffle=False)


# In[312]:


y_pred_test_nn = nn_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True')
plt.plot(y_pred_test_nn, label='NN')
plt.title("ANN's Prediction")
plt.xlabel('Observation')
plt.ylabel('Open Price')
plt.legend()
plt.show();


# In[313]:


reduction_nn_pred = scaler.inverse_transform(y_pred_test_nn)
reduction_test = df1.iloc[split_point:-1].copy()
reduction_y_test = reduction_test[predict_days:]


# In[314]:


print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_nn_pred))
print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_nn_pred))
print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_nn_pred)))
print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_nn_pred))


# # 台股、納斯達克、標普500多變量LSTM

# In[19]:


df1 = pd.read_csv("/Users/chenkaijung/Documents/大專生研究計畫/TW.csv")
df1.drop(['Close', 'High', 'Low', 'Close', 'Volume','Adj Close'], axis=1, inplace=True)
df1['Date'] = pd.to_datetime(df1['Date'])
df1 = df1.set_index(['Date'], drop=True)
df1 


# In[20]:


df2 = pd.read_csv("/Users/chenkaijung/Documents/大專生研究計畫/NASDAQ.csv")
df2.drop(['Close', 'High', 'Low', 'Close', 'Volume','Adj Close'], axis=1, inplace=True)
df2['Date'] = pd.to_datetime(df2['Date'])
df2 = df2.set_index(['Date'], drop=True)
df2


# In[21]:


df3 = pd.read_csv("/Users/chenkaijung/Documents/大專生研究計畫/S&P500.csv")
df3.drop(['Close', 'High', 'Low', 'Close', 'Volume','Adj Close'], axis=1, inplace=True)
df3['Date'] = pd.to_datetime(df3['Date'])
df3 = df3.set_index(['Date'], drop=True)
df3


# In[22]:


#資料歸一化
scaler = MinMaxScaler(feature_range=(-1, 1))

#需將資料做reshape的動作，使其shape為(資料長度,1) 

y_set= df3.values.reshape(-1,1)
sp500 = scaler.fit_transform(y_set)
sp500_df = pd.DataFrame(sp500)
y_set= df2.values.reshape(-1,1)
Nasdaq = scaler.fit_transform(y_set)
Nasdaq_df = pd.DataFrame(Nasdaq)
x_set= df1.values.reshape(-1,1)
TW = scaler.fit_transform(x_set)
TW_df = pd.DataFrame(TW)


# In[23]:


#合併兩個資料
result_df=pd.concat([TW_df , Nasdaq_df,sp500_df], axis = 1)
#去除nan值，暫用此方法，之後需重新整理資料
result_df = result_df.dropna()
result_df


# In[24]:


split_point = int(len(result_df)*0.9)
train = result_df.iloc[:split_point].copy()
test = result_df.iloc[split_point:-1].copy() #因為預測一天後的結果，最後一天預測並無解答，所以-1


# In[25]:


predict_days = 1
X_train = train[:-predict_days]
y_train = train[predict_days:]
X_test = test[:-predict_days]
y_test = test[predict_days:]


# In[26]:


X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values


# In[27]:


X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[28]:


lstm_model = Sequential()

lstm_model.add(LSTM(units=400, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(Dropout(0.25))

lstm_model.add(Dense(3))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history_lstm_model = lstm_model.fit(X_train, y_train, epochs=30, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])


# In[29]:


lstm_all = lstm_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.plot(y_test[:,0], label='TW_Open')
plt.plot(lstm_all[:,0][:,0], label='LSTM')
plt.title("LSTM's Prediction")
plt.xlabel('Observation')
plt.ylabel('Open')
plt.legend()
plt.show();


# In[30]:


reduction_LSTM_pred = scaler.inverse_transform(lstm_all[:,0])
reduction_test = df1.iloc[split_point:-1].copy()
reduction_y_test = reduction_test[predict_days:]


# In[31]:


print("Mean Absolute Error:", mean_absolute_error(reduction_y_test,reduction_LSTM_pred[:,0]))
print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_LSTM_pred[:,0]))
print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_LSTM_pred[:,0])))
print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_LSTM_pred[:,0]))


# In[24]:


lstm_model = Sequential()
lstm_model.add(LSTM(units=100, return_sequences=False, input_shape=(X_train.shape[1],1)))
lstm_model.add(Dropout(0.1))
#lstm_model.add(LSTM(units=50))

lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])


# In[25]:


y_pred_test_lstm = lstm_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.plot(y_test[:,0], label='TW_Open')
plt.plot(y_pred_test_lstm, label='LSTM')
plt.title("LSTM's Prediction")
plt.xlabel('Observation')
plt.ylabel('Open')
plt.legend()
plt.show();


# In[26]:


reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
reduction_test = df1.iloc[split_point:-1].copy()
reduction_y_test = reduction_test[predict_days:]
print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[38]:



for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=200, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.1))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[34]:



for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=200, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.2))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[13]:


for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=200, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.3))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[36]:


for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=200, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.4))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[37]:


for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=200, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.5))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[14]:


for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=300, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.1))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[32]:


for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=300, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.2))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[16]:


for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=300, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.3))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[31]:


for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=300, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.4))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[18]:


for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=300, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.5))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[19]:


for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=400, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.1))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[20]:


for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=400, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.2))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[21]:


for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=400, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.3))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[22]:


for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=400, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.4))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[23]:


for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=400, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.5))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[24]:


for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=500, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.1))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[25]:


for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=500, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.2))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[27]:


for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=500, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.3))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[28]:


for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=500, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.4))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[29]:


for i in range(1,6):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=500, return_sequences=False, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.5))
#lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, shuffle=True, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    reduction_Lstm_pred = scaler.inverse_transform(y_pred_test_lstm)
    reduction_test = df1.iloc[split_point:-1].copy()
    reduction_y_test = reduction_test[predict_days:]
    print("==================")
    print("Mean Absolute Error:", mean_absolute_error(reduction_y_test, reduction_Lstm_pred))
    print('Mean Squared Error:', mean_squared_error(reduction_y_test, reduction_Lstm_pred))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(reduction_y_test, reduction_Lstm_pred)))
    print("Coefficient of Determination:", r2_score(reduction_y_test, reduction_Lstm_pred))


# In[ ]:




