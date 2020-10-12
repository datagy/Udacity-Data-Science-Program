import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loads messages and categories data. Returns a dataframe, df."""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, on='id', how='outer')
    
    return df

def clean_data(df):
    """Splits categorical data into seperate onehotencoded columns.
    Renames columns.
    Removes duplicates."""
    
    categories = df['categories'].str.split(';', expand=True) # Split on ;
    row = categories.iloc[0, :] # Get first row
    category_colnames = [record[:-2] for record in row]  # list comprehension of category names
    categories.columns = category_colnames #rename categories columns

    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].apply(pd.to_numeric)

    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)

    df = df.drop_duplicates(keep='first')

    for col in df.columns[4:]:
        df[col] = df[col].replace(2, 1)

    return df


def save_data(df, database_filename):
    """Exports the dataframe to a SQLite database."""
    
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False)  


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