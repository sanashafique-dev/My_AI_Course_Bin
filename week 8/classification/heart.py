import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
 
df= pd.read_csv('week 8/classification/heart-2.csv')
print(df.head())
print(df.tail())
print(df.sample(5))
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df["target"].unique())
print(df["target"].value_counts())

#----------Seaborn-----------
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#----------Target Count-----------
sns.countplot(x="target", data=df)
plt.title("Heart Disease Count")
plt.show()

#----------Age vs Target------------
sns.scatterplot(x="age",y="target", data=df)
plt.title("Age vs Target")
plt.show()

#----------Cholesterol Distribution--------
sns.histplot(df["chol"], kde=True)
plt.title("Cholesterol Distribution")
plt.show()

#----------Age Distribution----------
sns.histplot(df["age"], kde=True)
plt.title("Age Distribution")
plt.show()

#----------Chest Pain Type----------
sns.countplot(x="cp", data=df)
plt.title("Chest Pain Type")
plt.show()

#----------- MACHINE LEARNING ----------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

#-------- Target and label -----------

X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#--------Standard Scaling---------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#---------Train Model---------
model = LogisticRegression()
model.fit(X_train, y_train)

#----------prediction ------------
y_pred = model.predict(X_test)

#--------Evaluation-----------

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :", accuracy)
precision = precision_score(y_test, y_pred)
print("Precision :", precision)
f1 = f1_score(y_test, y_pred)
print("F1 Score :", f1)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))