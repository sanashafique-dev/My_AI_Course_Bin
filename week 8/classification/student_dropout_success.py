import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import  seaborn as sns

df= pd.read_csv('week 8\classification\student_dropout_success.csv',sep=';')

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

print(df['Target'].unique())

print(df['Target'].value_counts())

#---------Seaborn---------------

plt.figure(figsize=(16,12))
sns.heatmap(df.corr(numeric_only=True),
            cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#---------Target Count---------

sns.countplot(x="Target", data=df)
plt.title("Student Status")
plt.show()

#----------Age Distribution---------

sns.histplot(df["Age at enrollment"], kde=True)
plt.title("Age Distribution")
plt.show()


#----------Admission Grade-----------

sns.histplot(df["Admission grade"], kde=True)

plt.title("Admission Grade")

plt.show()


#----------Previous Qualification Grade------------
sns.histplot(df["Previous qualification (grade)"], kde=True)

plt.show()

#-------------Age vs Admission Grade---------
sns.scatterplot(
    x="Age at enrollment",
    y="Admission grade",
    hue="Target",
    data=df
)

plt.show()

#----------Admission Grade vs Target-------------
sns.boxplot(
    x="Target",
    y="Admission grade",
    data=df
)

plt.show()

#---------Tuition Status------------
sns.countplot(
    x="Tuition fees up to date",
    hue="Target",
    data=df
)

plt.show()

#-----------Scholarship Holder-----------

sns.countplot(
    x="Scholarship holder",
    hue="Target",
    data=df
)

plt.show()

#---------- Machine learning ------------

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#-----------Label Encoding---------------

encoder = LabelEncoder()
df["Target"] = encoder.fit_transform(df["Target"])


#---------- feature and target ----------------
X = df.drop("Target", axis=1)

y = df["Target"]

#----------- Train-Test Split -------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#------------Standard Scaling----------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

#----------------Random forest -----------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

#----------prediction-------------

y_pred = model.predict(X_test)

#-------------Evaluation matrices-----------

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred, average="weighted"))
print("Recall :", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score :", f1_score(y_test, y_pred, average="weighted"))
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(classification_report(y_test, y_pred))



#-----------Confusion Matrix-------------
plt.figure(figsize=(7,6))

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