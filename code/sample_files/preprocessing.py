# Importing required packages
import pandas as pd

# Function to rectify the datatype of datetime column
def convert_to_datetime(column_name,df):
   df[column_name] = pd.to_datetime(df[column_name])
   return df[column_name]

# Function to preprocess 'Action' column
def clean_action_column(data):
    data['Action'] = data['Action'].apply(lambda x: 'read_reviews' if x in ('read_review') else x)
    data['Action'] = data['Action'].apply(lambda x: 'add_to_wishlist' if x in ('add_to_wishist') else x)
    return data

# Function for creating date features
def create_date_features(data, date_column):
    # Creating date level features
    data['Date'] = data[date_column].dt.date
    data['DayOfWeek'] = data[date_column].dt.dayofweek
    data['DayOfMonth'] = data[date_column].dt.day
    return data

# Function to retrieve target customers
# Target group of people are those who added items to the cart (Trigger point)
def get_target_customers(data):
    df_base = (data[data['Action']=='add_to_cart']
              .groupby('User_id').agg({'Category':'max','SubCategory':'max'}).reset_index())
    return df_base

# Function for filtering the dataset by max purchase date for each user (All users who have done add_to_cart event)
def filter_dataset_by_max_purchase_date(df,df_base):
    temp = (df.sort_values(by='DateTime')
        [(df.DateTime == df.User_id.map(df[df['Action']=='purchase'].groupby('User_id').DateTime.max()))]
       )
    purchase_users = temp[temp['Action']=='purchase'].groupby('User_id')['Action'].count().reset_index()
    df_base = pd.merge(df_base, purchase_users,on='User_id',how='left')
    df_base['Action'] = df_base['Action'].fillna(0)
    df_base.rename(columns={'Action':'Target'},inplace=True)
    return df_base

# Function to handle missing values
def handling_missing_values(df,column_name):
    df[column_name] = df[column_name].fillna(-1)
    df = df.fillna(0)
    return df