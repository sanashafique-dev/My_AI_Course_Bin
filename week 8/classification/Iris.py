import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('week 8\classification\Iris.csv')
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

#--------- Seaborn -----------
plt.figure(figsize=(8,6))

sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()

#----------Species Count----------
sns.countplot(x="Species", data=df)

plt.title("Species Count")

plt.show()

#--------Sepal Length Distribution-----------
sns.histplot(df["SepalLengthCm"], kde=True)

plt.title("Sepal Length Distribution")

plt.show()

#------------Petal Length Distribution----------
sns.histplot(df["PetalLengthCm"], kde=True)

plt.title("Petal Length Distribution")

plt.show()

#----------Sepal Length vs Sepal Width-----------
sns.scatterplot(
    x="SepalLengthCm",
    y="SepalWidthCm",
    hue="Species",
    data=df
)

plt.title("Sepal Length vs Sepal Width")

plt.show()

#---------Petal Length vs Petal Width----------
sns.scatterplot(
    x="PetalLengthCm",
    y="PetalWidthCm",
    hue="Species",
    data=df
)

plt.title("Petal Length vs Petal Width")
plt.show()

#------------Boxplot (Sepal Length)-----------
sns.boxplot(
    x="Species",
    y="SepalLengthCm",
    data=df
)

plt.title("Species vs Sepal Length")

plt.show()

#------------Violin Plot------------
sns.violinplot(
    x="Species",
    y="PetalWidthCm",
    data=df
)

plt.title("Species vs Petal Width")

plt.show()


#------------ Machine Learning ----------
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

df.drop("Id", axis=1, inplace=True)

#---------Label Encoding--------
encoder = LabelEncoder()

df["Species"] = encoder.fit_transform(df["Species"])

print(df.head())

#---------X and y-----------
X = df.drop("Species", axis=1)

y = df["Species"]
#----------Train-Test Split-------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#-----------Standard Scaling-----------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

#--------Train Model----------
model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#----------Metrics--------------
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred, average="weighted"))
print("Recall :", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score :", f1_score(y_test, y_pred, average="weighted"))
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(classification_report(y_test, y_pred))


#----------------Confusion Matrix Visualization----------
plt.figure(figsize=(6,5))

sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()