# Label Encoding and Preprocessing, Red Hat Business Value Challenge
# Note:  I will include the basic way in which one would go about creating X, y variables, 
# defining & fitting a random forest classifier (for simplicity, also our label encoder is designed
# best for those purposes, as other linear based models would have issues w/ the way in which the 
# data has been encoded.)

# importing libraries
import numpy as np
import pandas as pd
# from IPython.display import display, HTML (for displaying output in IPython Notebook)

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier


# reading in data; creating dataframes
# using the code below, as an example, people when defined as 'pd.read_csv('...')' creates a
# dataframe named 'people' w/ the data from said .csv
people = pd.read_csv('people.csv')
activity_train = pd.read_csv('act_train.csv')
activity_test = pd.read_csv('act_test.csv')


'''
py 2.7 = print people
py 3 = print (people)

i'll be using the 2.7 to make sure what im posting here doesn't have any errors
you'll want to swith to the v3 variation 

'''


# merging the dataframes into train, test
df = activity_train.merge(people, how='left', on='people_id' )
df_test = activity_test.merge(people, how='left', on='people_id' )


# want to take a look at the first rows of the new dataframe?
# df.head() or df_test.head() (the default is 5 rows; you can look at say, the first 20 using df_test.head(20)) 


# the shape of the dataframes
# print df.shape
# print df_test.shape


# filling NaN values first
df = df.fillna('0', axis=0)
df_test = df_test.fillna('0', axis=0)


# Preprocessing:


# a multi-column LabelEncoder()
# not sure if this is the most memory efficient as compared to other ways of dealing w/ encoding
# however a benefit of this function is that it would work with many other datasets and is versatile
# with some slight alterations to the 'columns'
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns 

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# defining a processor 
def processor(data):
    data = MultiColumnLabelEncoder(columns = ['people_id','activity_id', 'activity_category', 'date_x', 'char_1_x', 'char_2_x',
                                        'char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x', 'char_9_x',
                                        'char_10_x', 'char_1_y', 'group_1', 'char_2_y', 'date_y', 'char_3_y', 'char_4_y',
                                        'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y']).fit_transform(df)
    
    bool_map = {True:1, False:0}

    data = data.applymap(lambda x: bool_map.get(x,x))
    
    return data


# applying processor to training and test data
df_encoded = processor(df)
df_test_encoded = processor(df_test)


# a look at the new encoded dataframes
print df_encoded.head()
print df_test_encoded.head()

# we can see now that the datatypes have been changed to numeric
print df_encoded.dtypes
print df_test_encoded.dtypes


# Modeling:


# creating X, y variables (using .pop will now exclude 'outcome' (the target) from 'X')
X = df_encoded
y = X.pop('outcome')


# checking the shape once again
print X.shape
print y.shape


# i'm a little confused about how to approach this going forward, i think i'm overthinking it
# generally this is how i would proceed however
# random state 7, for reproducable results, test_size .25 meaning 'hold-out 25% of data as test set'
# we have a test set already, which is where my confusion arises from
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)


# random forest classifier (77 jobs is low but memory friendly; we'll do a grid search later anyways)
model = RandomForestClassifier(77, n_jobs=-1, random_state=7)


# fitting the model to training data
model.fit(X_train, y_train)


# scoring the model against test data
print "model score ", model.score(X_test, y_test)

# predicting test data
pred = model.predict(X_test)
print pred















