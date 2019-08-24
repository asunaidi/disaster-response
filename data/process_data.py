import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Description: This function is used to load and merge messages and categories datasets

    Arguments:
    messages_filepath: string. path of the csv file containing messages
    categories_filepath: string. path of the csv file containing categories

    Returns:
    A pandas DataFrame containing messgaes and categories
    """
    # Load messages
    messages = pd.read_csv(messages_filepath)
    # Load categories
    categories = pd.read_csv(categories_filepath)
    # Merge messages with categories
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Description: This function is used to  2:clean the dataset containing messages and categories

    Arguments:
    df: dataframe. Dataframe containing the merged messages and categories datasets.

    Returns:
    df: dataframe. Dataframe containing the input after cleaning the data 
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # this row is used to extract a list of new column names for categories.
    category_colnames = row.map(lambda x: x[:-2]).values  
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].map(lambda x: x[-1:])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # make sure we only have 1s & 0s in the values
    columns_to_clean = []
    for column in categories:      
        #remove columns with no variance
        if len(categories[column].unique()) < 2:
                categories.drop(column, axis = 1, inplace = True)
        elif len(categories[column].unique()) > 2:
            columns_to_clean.append(column)
            
    # drop the original categories column from `df`
    df = df.drop('categories', axis = 1 )
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    #remove rows that have values other than 1 or 0
    for column in columns_to_clean:
        df = df[(df[column] == 1) | (df[column] == 0)]
        
    #reomve duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Description: This function is used to  Save the clean dataset into an sqlite database

    Arguments:
    df: datafram. The clean data in Pandas Dataframe
    database_filename: string. the path of the file where the SQL database will be saved

    Returns: None
    """
    engine = create_engine('sqlite:///' + database_filename)

    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()