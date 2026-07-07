import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('week 8/Boston.csv',delimiter=',')

#-------------PANDAS------------

print(df.head(3))

print(df.tail(3))

print(df.info())

print(df.describe())

print(df.shape)

print(df.isnull().sum())

#------------   DATA VISUALIZATION  -------------

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

#------------   Rooms vs House Price   ------------

sns.scatterplot(x='rm',y='medv',data=df)
plt.title("Rooms vs House Price")
plt.show()

#------------   House Price Distribution  ----------

sns.histplot(df["medv"], kde=True)
plt.title("House Price Distribution")
plt.show()

#------------    MACHINE LEARNING  ------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

#------------ features and output--------------

x= df.drop('medv',axis=1)
y=df['medv']

# ------------train test split-------------------

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.20, random_state=42)

#------------standard scale---------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#-----------train model----------------

model = LinearRegression()
model.fit(X_train, y_train)
#------------prediction---------------

y_pred = model.predict(X_test)

#------------ Evaluation----------------

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

#----------plot graph-------------

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price (y_test)")
plt.ylabel("Predicted Price (y_pred)")
plt.title("Actual vs Predicted House Prices")

# -----ideal line (perfect prediction line)-----
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')

plt.show()
