import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('week 8/reggresion/housing[1].csv')
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated())
print(df['ocean_proximity'].unique())
print(df['ocean_proximity'].value_counts())
print(df.corr(numeric_only=True))

#-------------SEABORN-----------
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm')
plt.title("Heatmap ")
plt.show()

#------------Median Income vs House Value--------
sns.scatterplot(x="median_income",y="median_house_value",data=df)
plt.title("Median Income vs House Value")
plt.show()

#--------------Housing Age vs House Value------
sns.scatterplot( x="housing_median_age",y="median_house_value",data=df)
plt.title("Housing Age vs House Value")
plt.show()

#---------Population vs House Value---------
sns.scatterplot(x="population",y="median_house_value", data=df)
plt.title("Population vs House Value")
plt.show()

#-----------Ocean Proximity Count---------
plt.figure(figsize=(8,5))
sns.countplot( x="ocean_proximity", data=df)
plt.xticks(rotation=45)
plt.title("Ocean Proximity Count")
plt.show()

#---------House Value Distribution----------
sns.histplot(df["median_house_value"], kde=True)
plt.title("Median House Value Distribution")
plt.show()

#-----------Boxplot-----------
plt.figure(figsize=(10,6))
sns.boxplot(x="ocean_proximity", y="median_house_value",data=df)
plt.xticks(rotation=45)
plt.title("Ocean Proximity vs House Value")
plt.show()

df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].mean())

#------------ MACHINE LEARNING ---------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#-------Label encoding----------
encoder = LabelEncoder()
df["ocean_proximity"] = encoder.fit_transform(df["ocean_proximity"])
print(df.head())

#---------Feature and target -------------
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#------Standard Scaling ----------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#-------  Model -------------
model = LinearRegression()
model.fit(X_train, y_train)

#----------prediction-----------
y_pred = model.predict(X_test)

#-----------Evaluation-------------
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R2 Score :", r2_score(y_test, y_pred))

#----------Cofficient----------
coef = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print(coef)

#------------Actual vs Predicted Scatter Plot-------------
plt.figure(figsize=(7,6))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Price")
plt.show()