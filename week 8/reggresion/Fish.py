import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


#------------ pandas -----------
df= pd.read_csv("week 8/reggresion/Fish.csv", delimiter=',')

print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df['Species'].unique())
print(df['Species'].value_counts())

#-----------  seaborn ------------
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm')
plt.title('Correlation heatmap')
plt.show()

#-----------Species count---------------
sns.countplot(data=df,x='Species')
plt.title('Fish Species')
plt.show()

#---------Length3 vs Weight--------
sns.scatterplot(data=df, x="Length3", y="Weight")
plt.title("Length3 vs Weight")
plt.show()

#--------Height vs Weight---------
sns.scatterplot(data=df, x="Height", y="Weight")
plt.title("Height vs Weight")
plt.show()

#-----------Width vs Weight-------
sns.scatterplot(data=df, x="Width", y="Weight")
plt.title("Width vs Weight")
plt.show()

#--------Weight Distribution------
sns.histplot(df["Weight"], kde=True)
plt.title("Weight Distribution")
plt.show()

#----------- MACHINE LEARNING ------------
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



#----------Label Encoding----------

encoder = LabelEncoder()
df["Species"] = encoder.fit_transform(df["Species"])

# ----------Features and Target---------
X = df.drop("Weight", axis=1)
y = df["Weight"]

#------------ Train Test Split----------
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

#----------- Standard Scaling------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#----------- Model-----------
model = LinearRegression()

model.fit(X_train, y_train)

#----------- Prediction-----------
y_pred = model.predict(X_test)

#------------- Evaluation--------------
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R2 Score :", r2_score(y_test, y_pred))

#--------------Actual vs Predicted---------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)
plt.xlabel("Actual Weight")
plt.ylabel("Predicted Weight")
plt.title("Actual vs Predicted Weight")
plt.show()