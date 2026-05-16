import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
  
x=[1,2,3,4,5,6,7]
y=[20,40,10,90,30,120,60]
df=pd.DataFrame({'Days':x,'NO of Peoples':y})
print(df. head(7))

sns.lineplot(x='Days',y='NO of Peoples', data=df)
plt.show()
df=sns.load_dataset("tips")
print(df.head())

sns.lineplot(x='day',y='total_bill',data=df,hue='sex',style='time',palette='flare')
plt.show()
df=sns.load_dataset("exercise")
print(df.head())
sns.barplot(x='id',y='time',data=df,hue='kind',palette='spring',ci=False)
plt.show()