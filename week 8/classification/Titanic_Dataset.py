import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#---------Pandas-----------
df= pd.read_csv('week 8\classification\Titanic-Dataset.csv')
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


#----------Seaborn------------
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.show()

#-----------Survival Count------------
sns.countplot(x="Survived", data=df)

plt.title("Survival Count")

plt.show()

#---------Gender Count-----------
sns.countplot(x="Sex", data=df)

plt.title("Gender Distribution")

plt.show()

#----------Passenger Class------------
sns.countplot(x="Pclass", data=df)

plt.title("Passenger Class")

plt.show()

#---------Gender vs Survival-----------
sns.countplot(x="Sex",
              hue="Survived",
              data=df)

plt.title("Gender vs Survival")

plt.show()

#--------------Passenger Class vs Survival------------
sns.countplot(x="Pclass",
              hue="Survived",
              data=df)

plt.title("Passenger Class vs Survival")

plt.show()

#-----------Age Distribution---------------
sns.histplot(df["Age"], kde=True)

plt.title("Age Distribution")

plt.show()

#-------------Fare Distribution------------
sns.histplot(df["Fare"], kde=True)

plt.title("Fare Distribution")

plt.show()

#-----------Age vs Fare-----------
sns.scatterplot(x="Age",
                y="Fare",
                hue="Survived",
                data=df)

plt.show()


#-------------- Machince Learning ------------
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

#-----------Handle Missing Values-------------
df["Age"] = df["Age"].fillna(df["Age"].mean())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df["Cabin"] = df["Cabin"].fillna("Unknown")

#-----------Drop Unnecessary Columns-------------
df.drop(["PassengerId","Name","Ticket","Cabin"], axis=1, inplace=True)

#-----------Label Encoding------------
encoder = LabelEncoder()

df["Sex"] = encoder.fit_transform(df["Sex"])

df["Embarked"] = encoder.fit_transform(df["Embarked"])

#-----------X and y-------------
X = df.drop("Survived", axis=1)

y = df["Survived"]

#---------Train-Test Split----------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#------------Standard Scaling------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

#-----------Logistic Regression------------
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#------------Metrices------------

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred))
print("Recall :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(classification_report(y_test, y_pred))
#-------------Plot----------------

plt.figure(figsize=(6,5))

sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()