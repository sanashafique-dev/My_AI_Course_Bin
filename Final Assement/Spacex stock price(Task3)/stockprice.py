import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
#------------Pandas--------------
df= pd.read_csv('Final Assement/Spacex stock price(Task3)/spacex.csv')
print(df.head())
print(df.tail())
print(df.columns)
print(df.dtypes)
print(df.info())
print(df.shape)
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.max())

#--------------Numpy----------------

print("Maximum Price :", np.max(df['Close']))
print("Minimum Price :", np.min(df['Close']))
print("Average Close Price :", np.mean(df['Close']))
print("Average Close Price :", np.median(df['Close']))
print("Standard Deviation :", np.std(df['Close']))
print("Variance :", np.var(df['Close']))

#------------Data Visualization-----------

plt.figure(figsize=(10,6))
sns.lineplot(x='Open',y="Close",data=df)
plt.title('open and close stock price')
plt.xlabel("Open")
plt.ylabel('Close')
plt.savefig('Final Assement/Spacex stock price(Task3)/images/open_close.png', dpi=300, bbox_inches="tight")
plt.show()

#-----------Distribution of Closing Price----------
plt.figure(figsize=(10,6))
sns.histplot(df['Close'],bins=10,kde=True)
plt.title('Distribution of Closing Price')
plt.xlabel('Closing Price')
plt.ylabel('Frequency')
plt.savefig('Final Assement/Spacex stock price(Task3)/images/Distribution_of_Closing.png', dpi=300, bbox_inches="tight")
plt.show()

#----------Close Price Trend-----------

plt.figure(figsize=(10,6))
sns.lineplot(x='Datetime',y='Close',data=df)
plt.title('Close Price Over Time')
plt.savefig('Final Assement/Spacex stock price(Task3)/images/Close_PriceTime.png', dpi=300, bbox_inches="tight")
plt.show()

#----------Trading Volume Trend-----------
plt.figure(figsize=(10,6))
sns.lineplot(x='Datetime',y='Volume',data=df)
plt.title('Trading Volume')
plt.savefig('Final Assement/Spacex stock price(Task3)/images/Trading_Volume.png', dpi=300, bbox_inches="tight")
plt.show()

#----------Correlation Heatmap--------
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('Final Assement/Spacex stock price(Task3)/images/Correlation_Heatmap.png', dpi=300, bbox_inches="tight")
plt.show( )

#---------pairplot----------
sns.pairplot(df[["Open","High","Low","Close"]])
plt.savefig('Final Assement/Spacex stock price(Task3)/images/pairplot.png', dpi=300, bbox_inches="tight")
plt.show()
#------Box plot----------

plt.figure(figsize=(8,5))
sns.boxplot(data=df[["Open","High","Low","Close"]])
plt.title("Boxplot of Stock Prices")
plt.savefig('Final Assement/Spacex stock price(Task3)/images/Boxplot_of_Stock_Prices.png', dpi=300, bbox_inches="tight")
plt.show()

#---------Regression Plot----------

plt.figure(figsize=(7,5))
sns.regplot(x="Open", y="Close", data=df)
plt.title("Open vs Close Regression")
plt.savefig('Final Assement/Spacex stock price(Task3)/images/Open_vs_Close_Regression.png', dpi=300, bbox_inches="tight")
plt.show()


df['Datetime']=pd.to_datetime(df['Datetime'])
df=df.sort_values('Datetime')
df.set_index("Datetime", inplace=True)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#-------Feature and Target----------

x = df[['Open','High','Low','Volume']]
y = df['Close']

#----------------Feature scaling----------
from sklearn.preprocessing import MinMaxScaler

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(x)

y_scaled = scaler_y.fit_transform(y.values.reshape(-1,1))

#-----------Sequence Function --------
sequence_length = 10

X = []
Y = []

for i in range(sequence_length, len(X_scaled)):
    X.append(X_scaled[i-sequence_length:i])
    Y.append(y_scaled[i])

X = np.array(X)
Y = np.array(Y)

print(X.shape)
print(Y.shape)


#----------train test split------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_scaled,
    test_size=0.2,
    shuffle=False
)
X_train = X_train.reshape((X_train.shape[0],1,X_train.shape[1]))

X_test = X_test.reshape((X_test.shape[0],1,X_test.shape[1]))
print(X_train.shape)
print(X_test.shape)

#--------- DEEP LEARNING ------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN,LSTM,GRU,Dense,Dropout


#----------- RNN --------------

rnn_model=Sequential()
rnn_model.add(SimpleRNN(units=64,activation='tanh',return_sequences=False,input_shape=(X_train.shape[1], X_train.shape[2])))

rnn_model.add(Dropout(0.2))
rnn_model.add(Dense(32,activation='relu'))
rnn_model.add(Dense(1))

rnn_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
rnn_history = rnn_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=4,
    validation_data=(X_test,y_test),
    verbose=1
)

#----------------- LSTM ------------------
lstm_model=Sequential()
lstm_model.add(LSTM(units=64,return_sequences=True,input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=32,return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(16,activation='relu'))
lstm_model.add(Dense(1))
lstm_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

lstm_history = lstm_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=4,
    validation_data=(X_test,y_test),
    verbose=1
)


#-------------- GRU ---------------
gru_model = Sequential()

gru_model.add(GRU(units=64,return_sequences=True,input_shape=(X_train.shape[1], X_train.shape[2])))
gru_model.add(Dropout(0.2))
gru_model.add(GRU(units=32))
gru_model.add(Dropout(0.2))
gru_model.add(Dense(16,activation='relu'))
gru_model.add(Dense(1))

gru_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

gru_history = gru_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=4,
    validation_data=(X_test,y_test),
    verbose=1
)

#------------- Predictions -----------
rnn_pred = rnn_model.predict(X_test)
lstm_pred = lstm_model.predict(X_test)
gru_pred = gru_model.predict(X_test)

# Inverse Transform

rnn_pred = scaler_y.inverse_transform(rnn_pred)
lstm_pred = scaler_y.inverse_transform(lstm_pred)
gru_pred = scaler_y.inverse_transform(gru_pred)
y_actual = scaler_y.inverse_transform(y_test)


#-------- Evaluation -----------
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

print("--------------- RNN ----------------")
print("MAE :", mean_absolute_error(y_actual, rnn_pred))
print("MSE :", mean_squared_error(y_actual, rnn_pred))
print('R2:',r2_score(y_actual,rnn_pred))
print()

print("--------------- LSTM ----------------")
print("MAE :", mean_absolute_error(y_actual, lstm_pred))
print("MSE :", mean_squared_error(y_actual, lstm_pred))
print('R2:',r2_score(y_actual,lstm_pred))
print()
print("--------------- GRU ----------------")
print("MAE :", mean_absolute_error(y_actual, gru_pred))
print("MSE :", mean_squared_error(y_actual, gru_pred))
print('R2:',r2_score(y_actual,gru_pred))

#----------Best Model ------------
rnn_mae = mean_absolute_error(y_actual, rnn_pred)
lstm_mae = mean_absolute_error(y_actual, lstm_pred)
gru_mae = mean_absolute_error(y_actual, gru_pred)

models = {
    "RNN": rnn_mae,
    "LSTM": lstm_mae,
    "GRU": gru_mae
}

best_model = min(models, key=models.get)

print("\n===================================")
print("Best Model :", best_model)
print("Lowest MAE :", models[best_model])
print("===================================")


#------------ Plot Prediction -----------

plt.figure(figsize=(14,6))
plt.plot(y_actual, label="Actual", color="black", linewidth=3)
plt.plot(rnn_pred, label="RNN Prediction")
plt.plot(lstm_pred, label="LSTM Prediction")
plt.plot(gru_pred, label="GRU Prediction")

plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.savefig('Final Assement/Spacex stock price(Task3)/images/Actual_vs_PredictedStock.png', dpi=300, bbox_inches="tight")
plt.show()


#-------------- Compare  loss curves------------
plt.figure(figsize=(12,6))

# RNN
plt.plot(rnn_history.history['loss'], label='RNN Train')
plt.plot(rnn_history.history['val_loss'], label='RNN Validation')

# LSTM
plt.plot(lstm_history.history['loss'], label='LSTM Train')
plt.plot(lstm_history.history['val_loss'], label='LSTM Validation')

# GRU
plt.plot(gru_history.history['loss'], label='GRU Train')
plt.plot(gru_history.history['val_loss'], label='GRU Validation')

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Final Assement/Spacex stock price(Task3)/images/Training_vs_ValidationLoss.png', dpi=300, bbox_inches="tight")
plt.show()

#---------- Model Summary ----------
print("----------- RNN Model -----------")
rnn_model.summary()
print("----------- LSTM Model -----------")
lstm_model.summary()
print("----------- GRU Model -----------")
gru_model.summary()