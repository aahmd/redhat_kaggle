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


# applying processor to training data
df_encoded = processor(df)


# a look at the new encoded dataframe
print df_encoded.head()



