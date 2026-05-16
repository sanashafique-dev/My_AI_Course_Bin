import pandas as pd

print(pd.__version__)

df_index=pd.read_csv('week4/FastFoodRestaurants.csv',delimiter=',')

print(df_index)

print("datatypes",df_index.dtypes)

print('df.info',df_index.info())

print('describes shows all the  statistics functions\n',df_index.describe())  # summary of all statistics functions

print('countingthe rows and column in dataframe',df_index.shape)  # no of rows and column

#last 2 rows

print('Last three Rows:')
print(df_index.tail(3))

#First two rows

print('First Three Rows:')
print(df_index.head(3))

#unique values  and count 

print(df_index['name'].unique())
print(df_index['name'].value_counts())


#all columns names

print('acess all column name',df_index.columns)

#access single col

city=df_index["city"]
print("acess the name column",city)

#multiple col

city_country=df_index[["city","country"]]
print("acess multiple colums:",city_country)

                    #use .loc

#single row using loc 

one_row=df_index.loc[[2]]
print("selecting single rows using loc:",one_row)

 # multiple row  using loc
two_rows=df_index.loc[[2,5]]
print("selecting multiple rows using loc:",two_rows)

#slice  of rows
slice_rows=df_index.loc[3:7]
print("slice of rows:",slice_rows)

#conditional selection of rows
select_rows=df_index.loc[df_index["city"]=="Saluda"]
print("selecting a sindle colum city  those value Saluda:",select_rows)

#select single col
select_rows2=df_index.loc[:2,'country']
print(select_rows2)

#select multiple colums
select_rows3=df_index.loc[:3,["name","postalCode"]]
print("select a slice of col:",select_rows3)

#select slice of column
select_rows4=df_index.loc[:2,"address":"name"]
print("select a slice of column",select_rows4)

#select combine row and col
select_rows5=df_index.loc[df_index["city"]=='Englewood',"country":"US"]
print("combined rows and col:",select_rows5)


#-----------------------using .iloc---------------------
select_row_iloc=df_index.iloc[[0, 4,6]]
print("select rows using iloc",select_row_iloc)


#select slice
select_row_iloc2=df_index.iloc[2:5]
print("select slice of row",select_row_iloc2)



#select combined row and column
select_row_iloc3=df_index.iloc[[1,3],2:7]
print("combined rows and column:",select_row_iloc3)


#select slice of col
select_row_iloc4=df_index.iloc[:,2:6]
print("select row and column",select_row_iloc4)

# MANIPULATION

df_index.loc[len(df_index.index)]=["401 N Jennings St","Saluda","US","us/sc/saluda/401njenningsst/-1161002137",34.00598,-81.7704,"McDonald's",29138,"SC","http://www.mcdonalds.com"]
print("modified data frame:")
print(df_index)

                  # Remove rows and column


# delete row 

df_index.drop(3, axis=0, inplace=True)

# delete row with index 2
df_index.drop(index=2, inplace=True)

# delete rows with index 3 and 5
df_index.drop([1, 5], axis=0, inplace=True)

print("Modified DataFrame - Remove Rows:",df_index)



# delete  column

df_index.drop('province', axis=1, inplace=True)
# delete marital status column
df_index.drop(columns='longitude', inplace=True)
# delete height and profession columns
df_index.drop(['websites ', 'latitude'], axis=1, inplace=True)
# display the modified DataFrame after deleting rows
print("Modified DataFrame  delete  columns:",df_index)

# -----------rename coloumn-----------
df_index.rename(columns={"city":"city_name"},inplace=True)
df_index.rename(mapper={"longitude":"longitude_changed","latitude":"latitude_changed"},axis=1,inplace=True)
print(df_index)

df_index.rename(index={0: 3}, inplace=True)
df_index.rename(mapper={1: 1, 2: 10}, axis=0, inplace=True)
print(df_index)

#--------- select rows using query---------

selected_rows=df_index.query('city=="Massena" or name=="OMG! Rotisserie"')
print(selected_rows. to_string())
print(len(selected_rows))