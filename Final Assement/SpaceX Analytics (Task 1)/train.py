import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#------------PANDAS----------

df= pd.read_csv("Final Assement/SpaceX Analytics (Task 1)/Space_Industry_Analytics_2010_2024.csv")

print(df)
#..........DATA UNDERSTANDING........

print(df.info())# col and datatypes

print(df.dtypes) # datatypes

print(df.describe()) #all statistic functions

print(df.shape)# number of rows and column

print(df.isnull().sum())#null missing values 

print(df.duplicated().sum()) # duplicate values

#----------- Data visualization---------------

#----------lanches vs company---------

plt.figure(figsize=(8,5))
sns.lineplot(data=df,
             x='Year', 
             y='Launches', 
             hue='Company')
plt.title('lanches vs company')
plt.savefig("Final Assement/SpaceX Analytics (Task 1)/imageslaunches_vs_company.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
#'''SpaceX shows rapid growth in launch activity after 2017.'''

#---------Revenue Growth Over Years-------

plt.figure(figsize=(8,5))
sns.lineplot(data=df, 
             x='Year',
            y='Revenue_USD_M', 
            hue='Company')
plt.title('Revenue Growth Over Years')
plt.savefig("Final Assement/SpaceX Analytics (Task 1)/images/revenue_growth.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
'''SpaceX demonstrates exceptional revenue growth.
   Rocket Lab shows gradual growth.
   NASA records zero commercial revenue'''

#----------Success Rate Comparison--------

plt.figure(figsize=(8,5))
sns.barplot(x="Company",
            y="Success_Rate_%",
            data=df)
plt.title('Success Rate Comparison')
plt.savefig("Final Assement/SpaceX Analytics (Task 1)/images/success_rate.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
'''NASA maintains a near-perfect success rate.'''

#---------Funding vs Revenue----------

plt.figure(figsize=(8,5))
sns.scatterplot(
    x="Budget_Funding_USD_M",
    y="Revenue_USD_M",
    hue="Company",
    data=df
)

plt.title("Funding vs Revenue")
plt.savefig("Final Assement/SpaceX Analytics (Task 1)/images/funding_vs_revenue.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#---------Employees vs Revenue---------

plt.figure(figsize=(8,5))
sns.scatterplot(x='Employees',
                y='Revenue_USD_M',
                hue='Company',
                data=df)
plt.title('Employees vs Revenue')
plt.savefig("Final Assement/SpaceX Analytics (Task 1)/images/employees_vs_revenue.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#--------Launches vs Revenue----------

plt.figure(figsize=(8,5))
sns.scatterplot(x='Launches',
                y='Revenue_USD_M',
                hue='Company',
                data=df)
plt.title('Launches vs Revenue')
plt.savefig("Final Assement/SpaceX Analytics (Task 1)/images/launches_vs_revenue.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#--------Correlation Heatmap--------

plt.figure(figsize=(8,5))
sns.heatmap( df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")
plt.savefig("Final Assement/SpaceX Analytics (Task 1)/images/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#---------MACHINE LEARNING---------

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

#-------- ENCODING-------------

le = LabelEncoder()

df["Company"] = le.fit_transform(df["Company"])

print(df.head())

#-----------FEATURES AND TARGET---------

x=df.drop('Revenue_USD_M',axis=1)

y = df["Revenue_USD_M"]

#-----------traintest------------
 
X_train, X_test, Y_train, Y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

#-----------LINEAR REGRESSION-----------

lr= LinearRegression()
lr.fit(X_train,Y_train)
prediction=lr.predict(X_test)

print("Liner Regession")
print("MAE",mean_absolute_error(Y_test,prediction))
lr_r2=r2_score(Y_test,prediction)
print('R2 Score:', lr_r2)

#------RANDOM FOREST-----------

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(X_train,Y_train)
rf_prediction=rf.predict(X_test)

print('Random Forest')
print('MAE:',mean_absolute_error(Y_test,rf_prediction))
rf_r2=r2_score(Y_test,rf_prediction)
print('R2 Score:',rf_r2)

#-----------MODEL COMPARISON-----------

if rf_r2>lr_r2:
    print('random forest performed better .')
else:
    print('linear regression performed better')

#-----------ACTUAL VS PREDICTED GRAPH-----------
plt.figure(figsize=(6,6))
plt.scatter(Y_test, prediction)
plt.title("Actual vs Predicted Revenue")

plt.savefig("Final Assement/SpaceX Analytics (Task 1)/images/actual_vs_predicted.png",dpi=300,bbox_inches="tight")

plt.show()
plt.close()