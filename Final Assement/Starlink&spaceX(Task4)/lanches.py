import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report,roc_curve,auc


df= pd.read_csv('Final Assement/Starlink&spaceX(Task4)/spacex_launches.csv')

print(df.head())

print(df.tail())

print(df.info())

print(df.describe())

print(df.shape)

print(df.columns)

print(df.dtypes)

print(df.isnull().sum())

print(df.duplicated().sum())

#---------Launch Success Distribution------------
plt.figure(figsize=(6,5))
sns.countplot(data=df, x="success")
plt.title("Launch Success Distribution")
plt.xlabel("Launch Result")
plt.ylabel("Number of Launches")
plt.xticks([0,1],["Failure","Success"])

plt.savefig("Final Assement/Starlink&spaceX(Task4)/images/Launch_Success_Distribution.png",dpi=300,bbox_inches="tight")
plt.show()

#-------------Rocket Usage--------------
plt.figure(figsize=(10,5))

rocket_counts = df["rocket_name"].value_counts()
sns.barplot(x=rocket_counts.values,y=rocket_counts.index)
plt.title("Rocket Usage")
plt.xlabel("Number of Launches")
plt.ylabel("Rocket")

plt.savefig("Final Assement/Starlink&spaceX(Task4)/images/Rocket_Usage.png",dpi=300,bbox_inches="tight")
plt.show()

#---------------Launchpad Usage-----------
plt.figure(figsize=(10,5))
launchpad_counts = df["launchpad_name"].value_counts()
sns.barplot(x=launchpad_counts.values,y=launchpad_counts.index)
plt.title("Launchpad Usage")
plt.xlabel("Launch Count")
plt.savefig("Final Assement/Starlink&spaceX(Task4)/images/Launchpad_Usage.png",dpi=300,bbox_inches="tight")
plt.show()

#------------Launches Per Year-----------

df["date_utc"] = pd.to_datetime(df["date_utc"])
launches = df.groupby(df["date_utc"].dt.year).size()
plt.figure(figsize=(12,5))

plt.plot(launches.index,launches.values,marker="o")

plt.title("Launches Per Year")
plt.xlabel("Year")
plt.ylabel("Total Launches")

plt.grid(True)
plt.savefig("Final Assement/Starlink&spaceX(Task4)/images/Launches_Per_Year.png",dpi=300,bbox_inches="tight")
plt.show()

#--------------Landing Success--------------
plt.figure(figsize=(6,5))
sns.countplot(data=df,x="landing_success")
plt.title("Landing Success")
plt.xticks([0,1],["Failed","Successful"])
plt.savefig("Final Assement/Starlink&spaceX(Task4)/images/Landing_Success.png",dpi=300,bbox_inches="tight")
plt.show()

#---------------Landing Types---------------
plt.figure(figsize=(10,5))
sns.countplot(data=df,y="landing_type",order=df["landing_type"].value_counts().index)
plt.title("Landing Types")
plt.savefig("Final Assement/Starlink&spaceX(Task4)/images/Landing_Types.png",dpi=300,bbox_inches="tight")

plt.show()

#---------------Payload Distribution------------
plt.figure(figsize=(8,5))
sns.histplot(df["payloads_count"],bins=10,kde=True)
plt.title("Payload Distribution")
plt.xlabel("Payload Count")
plt.savefig("Final Assement/Starlink&spaceX(Task4)/images/Payload_Distribution.png",dpi=300,bbox_inches="tight")

plt.show()

#-----------------Core Reuse-------------
plt.figure(figsize=(7,5))
sns.countplot(data=df,x="cores_reused")
plt.title("Rocket Core Reuse")
plt.savefig("Final Assement/Starlink&spaceX(Task4)/images/Core_Reuse.png",dpi=300,bbox_inches="tight")

plt.show()

#------------Correlation Heatmap-----------
plt.figure(figsize=(10,7))
sns.heatmap(df.select_dtypes(include="number").corr(),annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("Final Assement/Starlink&spaceX(Task4)/images/Correlation_Heatmap.png",dpi=300,bbox_inches="tight")

plt.show()

#-----------Rocket Success Rate-----------
rocket_success = df.groupby("rocket_name")["success"].mean()*100
plt.figure(figsize=(10,5))
rocket_success.plot(kind="bar")
plt.ylabel("Success Rate (%)")
plt.title("Rocket Success Rate")
plt.savefig("Final Assement/Starlink&spaceX(Task4)/images/Rocket_Success_Rate.png",dpi=300,bbox_inches="tight")

plt.show()

#--------------Launchpad Success Rate----------
launchpad_success = df.groupby("launchpad_name")["success"].mean()*100
plt.figure(figsize=(10,5))
launchpad_success.plot(kind="bar")
plt.ylabel("Success Rate (%)")
plt.title("Launchpad Success Rate")
plt.savefig("Final Assement/Starlink&spaceX(Task4)/images/Launchpad_Success_Rate.png",dpi=300,bbox_inches="tight")

plt.show()

#---------Pairplot------------
sns.pairplot(
df[[
"payloads_count",
"cores_reused",
"landing_success",
"success"]],
hue="success")
plt.savefig("Final Assement/Starlink&spaceX(Task4)/images/Pairplot.png",dpi=300,bbox_inches="tight")

plt.show()

df["landing_type"] = df["landing_type"].fillna("unknown")
df["details"] = df["details"].fillna("No Details")
print(df.isnull().sum())


#---------Encoding ----------------

encoder = LabelEncoder()

categorical = [
"rocket_name",
"launchpad_name",
"landing_type"
]

for col in categorical:

    df[col]=encoder.fit_transform(df[col].astype(str))

#------------Target and feature --------------
df.fillna(0, inplace=True)

X = df.drop(columns=[
    "success",
    "flight_number",
    "name",
    "rocket_id",
    "launchpad_id",
    "details",
    "date_utc"
])

y = df["success"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#----------Sequence Creation-----------
sequence_length = 5
X_sequences = []
y_sequences = []

for i in range(sequence_length, len(X_scaled)):
    X_sequences.append(X_scaled[i-sequence_length:i])
    y_sequences.append(y.iloc[i])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

print(X_sequences.shape)
print(y_sequences.shape)

#-------------Train test ---------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_sequences,
    y_sequences,
    test_size=0.2,
    shuffle=False
)

print(X_train.shape)
print(X_test.shape)

#---------------- RNN --------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

rnn_model = Sequential()

rnn_model.add(SimpleRNN( 64,input_shape=(X_train.shape[1], X_train.shape[2])))
rnn_model.add(Dropout(0.2))
rnn_model.add(Dense(32, activation="relu"))
rnn_model.add(Dense(1, activation="sigmoid"))

rnn_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

rnn_model.summary()
rnn_history = rnn_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=1
)


#------------- LSTM ----------------

from tensorflow.keras.layers import LSTM, Dense, Dropout

lstm_model = Sequential()

lstm_model.add(LSTM(units=64,return_sequences=True,input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=32,return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(16, activation="relu"))
lstm_model.add(Dense(1, activation="sigmoid"))

lstm_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

lstm_model.summary()
lstm_history = lstm_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=1
)

#-------------GRU--------------
from tensorflow.keras.layers import GRU

gru_model = Sequential()

gru_model.add(GRU( units=64, return_sequences=True,input_shape=(X_train.shape[1], X_train.shape[2])))
gru_model.add(Dropout(0.2))
gru_model.add(GRU(units=32))
gru_model.add(Dropout(0.2))
gru_model.add(Dense(16, activation="relu"))
gru_model.add(Dense(1, activation="sigmoid"))

gru_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

gru_model.summary()
gru_history = gru_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=1
)
#------------ Predictions ------------

rnn_pred = rnn_model.predict(X_test)
lstm_pred = lstm_model.predict(X_test)
gru_pred = gru_model.predict(X_test)
rnn_pred_class = (rnn_pred > 0.5).astype(int)

lstm_pred_class = (lstm_pred > 0.5).astype(int)

gru_pred_class = (gru_pred > 0.5).astype(int)
print("========== RNN ==========")

print("Accuracy :", accuracy_score(y_test, rnn_pred_class))
print("Precision :", precision_score(y_test, rnn_pred_class))
print("Recall :", recall_score(y_test, rnn_pred_class))
print("F1 Score :", f1_score(y_test, rnn_pred_class))
print(classification_report(y_test, rnn_pred_class))
print()

print("========== LSTM ==========")

print("Accuracy :", accuracy_score(y_test, lstm_pred_class))
print("Precision :", precision_score(y_test, lstm_pred_class))
print("Recall :", recall_score(y_test, lstm_pred_class))
print("F1 Score :", f1_score(y_test, lstm_pred_class))
print(classification_report(y_test, lstm_pred_class))
print()

print("========== GRU ==========")

print("Accuracy :", accuracy_score(y_test, gru_pred_class))
print("Precision :", precision_score(y_test, gru_pred_class))
print("Recall :", recall_score(y_test, gru_pred_class))
print("F1 Score :", f1_score(y_test, gru_pred_class))
print(classification_report(y_test, gru_pred_class))
#-------------CONFUSION MARIX OF RNN-------------

cm = confusion_matrix(y_test, rnn_pred_class)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True,fmt="d", cmap="Blues",xticklabels=["Failure","Success"],yticklabels=["Failure","Success"])
plt.title("RNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Final Assement/Starlink&spaceX(Task4)/images/RNN_Confusion_Matrix.png",dpi=300,bbox_inches="tight")
plt.show()

#------------CONFUSION MATRIX OF LSTM-----------------

cm = confusion_matrix(y_test, lstm_pred_class)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True,fmt="d", cmap="Blues",xticklabels=["Failure","Success"],yticklabels=["Failure","Success"])

plt.title("LSTM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Final Assement/Starlink&spaceX(Task4)/images/LSTM_Confusion_Matrix.png",dpi=300,bbox_inches="tight")
plt.show()

#-------------CONFUSION MATRIX OF GRU-----------

cm = confusion_matrix(y_test, gru_pred_class)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True,fmt="d", cmap="Blues",xticklabels=["Failure","Success"],yticklabels=["Failure","Success"])

plt.title("GRU Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Final Assement/Starlink&spaceX(Task4)/images/GRU_Confusion_Matrix.png",dpi=300,bbox_inches="tight")

plt.show()
#-------------- BEST MODEL ----------
rnn_acc = accuracy_score(y_test, rnn_pred_class)
lstm_acc = accuracy_score(y_test, lstm_pred_class)
gru_acc = accuracy_score(y_test, gru_pred_class)

models_score = {
    "RNN": rnn_acc,
    "LSTM": lstm_acc,
    "GRU": gru_acc
}

best_model = max(models_score, key=models_score.get)
#----------------TRAING VS VALIDATION ---------------
plt.figure(figsize=(12,6))

plt.plot(rnn_history.history["accuracy"],label="RNN Train")
plt.plot(rnn_history.history["val_accuracy"],label="RNN Validation")

plt.plot(lstm_history.history["accuracy"],label="LSTM Train")
plt.plot(lstm_history.history["val_accuracy"],label="LSTM Validation")

plt.plot(gru_history.history["accuracy"],label="GRU Train")
plt.plot(gru_history.history["val_accuracy"],label="GRU Validation")

plt.title("Training vs Validation Accuracy")

plt.legend()

plt.savefig(
"Final Assement/Starlink&spaceX(Task4)/images/Training_vs_Validation_Accuracy.png",
dpi=300,bbox_inches="tight")
plt.show()
#----------Accuracy Comparison Graph------------

models_names = ["RNN","LSTM","GRU"]
accuracy = [rnn_acc,lstm_acc,gru_acc]
plt.figure(figsize=(7,5))
sns.barplot(x=models_names,y=accuracy)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

plt.savefig("Final Assement/Starlink&spaceX(Task4)/images/Model_Accuracy_Comparison.png",dpi=300,bbox_inches="tight")
plt.show()


print("\n==============================")

print("Best Model :", best_model)

print("Highest Accuracy :", models_score[best_model])

print("==============================")