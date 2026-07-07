import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
df =pd.read_csv('week 8/advertising.csv', delimiter=',')

#-----------PANDAS------------

print(df.head(3))

print(df.tail(3))

print(df.info())

print(df.describe())

print(df.isnull().sum())

#--------- DATA VISUALIZATION ------------

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.title('correlation between features and sale ')
plt.show()

#---------- TV ADVERTISMENT VS SALES ---------

sns.scatterplot(x='TV',y='Sales',data=df)
plt.title('TV ADVERTISMENT VS SALES')
plt.show()

#--------- Radio ADVERTISMENT VS SALES --------

sns.scatterplot(x='Radio',y='Sales',data=df)
plt.title('Radio ADVERTISMENT VS SALES')
plt.show()

#--------- Newspaper ADVERTISMENT VS SALES -------

sns.scatterplot(x='Newspaper',y='Sales',data=df)
plt.title('Newspaper ADVERTISMENT VS SALES')
plt.show()

#-------- Sales Distribution -------------

sns.histplot(df['Sales'],kde=True)
plt.title('Sales Distribution')
plt.show()

#-------------- MACHINE LEARNING --------------

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#------------ FEATURES AND OUTPUT  ------------

x= df.drop('Sales',axis=1)
y=df['Sales']

#------------ TRAIN TEST  -------------
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=42)


#------------- STANDARD SCALE ---------
scaler = StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#------------ MODEL TRAIN ----------

model =LinearRegression()
model.fit(X_train,y_train)

#---------- PREDICTION -----------

y_Pred=model.predict(X_test)

#---------- EVALUATION -----------

print("MAE :", mean_absolute_error(y_test, y_Pred))

print("MSE :", mean_squared_error(y_test, y_Pred))

print("R2 Score :", r2_score(y_test, y_Pred))

#----------- COFFICIENT --------------

coef_df = pd.DataFrame(
    model.coef_,
    x.columns,
    columns=["Coefficient"]
)

print(coef_df)

#-------- FINAL PLOT -------------
plt.scatter(y_test, y_Pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

#-------- perfect prediction line -------- 
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.show()