import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#--------PANDAS--------
df=pd.read_csv('week 8/pakistan_social_media_usage_2026.csv',delimiter=',')

print(df.head(5))#first 5 rows

print(df.tail(3))# last three rows

print(df.info())# col and datatypes

print(df.describe()) #all statistic functions


# -----------Data visualization using seaborn--------

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()

# ---------count gender----------
sns.countplot(x='gender',data=df)
plt.show()

# -------- social media usage---------

sns.countplot(x='platform',data=df)
plt.title('social media usage')
plt.show()

#------- studyhour vs exam score-------

sns.scatterplot(x='study_hours',y='exam_score',data=df)
plt.show()

#-------- daily  social media usage vs examscore-----

sns.scatterplot(x='daily_hours',y='exam_score',data=df)
plt.show()

#-------mental health vs exam_score------

sns.lineplot(x='mental_health_rating',y='exam_score', data=df)
plt.show()

#-------examscore distribute by gender------

sns.boxplot(x='gender',y='exam_score',data=df)
plt.show()


#-------------    Machinelearning    -------------


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#-----data encode-----

le=LabelEncoder()
df['gender']=le.fit_transform(df['gender'])
df['platform']=le.fit_transform(df['platform'])

print('Encode data',df.head())

#-------write csv-----------
#df.to_csv('week 8/pakistan_social_media_usage_2026.csv',index=False)

#-----features and target------

x=df.drop('exam_score',axis=1)
y=df['exam_score']


#------train test--------

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

#----prediction---
y_pred = model.predict(X_test)


# ------evaluation------
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nMean Absolute Error:",mae)
print("\nMean Squared Error:",mse)
print("\nR2 Score:",r2)


plt.scatter(y_test, y_pred)

plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")

plt.show()

#----sample test----
sample = [[
    18,
    1,
    2,
    4.5,
    7,
    1200,
    2.5,
    6
]]

prediction = model.predict(sample)
print(prediction)
