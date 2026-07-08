import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('week 8/classification/data-2.csv')
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
print(df["diagnosis"].unique())
print(df["diagnosis"].value_counts())

#--------Seaborn----------
plt.figure(figsize=(18,12))

sns.heatmap(df.corr(numeric_only=True),
            cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.show()

#-----------Diagnosis Count-----------
sns.countplot(x="diagnosis", data=df)

plt.title("Diagnosis Count")

plt.show()

#---------Radius Mean Distribution--------
sns.histplot(df["radius_mean"], kde=True)

plt.title("Radius Mean Distribution")

plt.show()

#-------Texture Mean Distribution--------
sns.histplot(df["texture_mean"], kde=True)

plt.title("Texture Mean Distribution")

plt.show()

#----------Radius Mean vs Diagnosis----------
sns.boxplot(x="diagnosis",
            y="radius_mean",
            data=df)

plt.title("Radius Mean vs Diagnosis")

plt.show()

#-----------Area Mean vs Diagnosis--------
sns.boxplot(x="diagnosis",
            y="area_mean",
            data=df)

plt.title("Area Mean vs Diagnosis")

plt.show()

#-----------Radius Mean vs Area Mean-----------
sns.scatterplot(
    x="radius_mean",
    y="area_mean",
    hue="diagnosis",
    data=df
)

plt.title("Radius Mean vs Area Mean")

plt.show()

#----------Texture Mean vs Perimeter Mean--------
sns.scatterplot(
    x="texture_mean",
    y="perimeter_mean",
    hue="diagnosis",
    data=df
)

plt.show()

#----------- MACHINE LEARNING ------------

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
#----------Label Encoding -------
encoder = LabelEncoder()

df["diagnosis"] = encoder.fit_transform(df["diagnosis"])

print(df.head())

#-----------X and y-----------
X = df.drop("diagnosis", axis=1)

y = df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#-------------Standard Scaling----------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

#-----------Train Model----------
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

#----------prediction-----------
y_pred = model.predict(X_test)

#-----------Metrics------------
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred))
print("Recall :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(classification_report(y_test, y_pred))

#-----------Confusion Matrix---------
plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()