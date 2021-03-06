import sys, re, pickle, nltk
nltk.download(['punkt', 'wordnet'])
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath):
    """Takes the database_filepath as an input and loads the data. 
    
    Returns:
    X - message data
    y - target values
    category_names - names of category headers as a list"""
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages', engine) #reads in SQL table, without using a pd_read_sql query
    X = df.message # generates (1,0) array for message to be transformed and processed via TF-IDF
    y = df.drop(['id', 'message', 'genre', 'original'], axis=1) #Keeps only the categories
    category_names = y.columns.tolist() #generates category names
    
    return X, y, category_names


def tokenize(text):
    """Takes text as an input and returns tokenized text.
    -Removes URLs and replaces with urlplaceholder
    - keeps only text
    - lemmatizes words
    - strips whitespace
    
    Returns clean tokens"""    
    
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    text = re.sub(r'[^\w\s]','',text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens

def build_model():
    """Builds a pipeline, inserts parameters, and returns a cv."""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),  #Convert a collection of text documents to a matrix of token counts
        ('tfidf', TfidfTransformer()),  #Transform a count matrix to a normalized tf or tf-idf representation
        ('clf', MultiOutputClassifier(RandomForestClassifier()))  # set the weights of classifiers and training the data sample in each iteration such that it ensures the accurate predictions of unusual observations
    ])
    parameters = {
#         'vect__ngram_range': ((1, 1), (1, 2)),  #(1,1) uses only unigrams (single words), (1,2) identifies unigrams and bigrams
        'vect__max_df': (0.5, 1.0),  #ignore words that occur in 50%, 75% or 100% of documents
        'tfidf__use_idf': (True, False) # determines whether idf (TRUE) is used or only tf (FALSE)
        }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=4, cv=3)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the mode and returns as classification report"""
#     Y_pred = model.predict(X_test)
#     print(classification_report(Y_test, Y_pred, target_names = category_names))
#     print('---------------------------------')
#     for i in range(Y_test.shape[1]):
#         print('%25s accuracy : %.2f' %(category_names[i], accuracy_score(Y_test[:,i], Y_pred[:,i])))
    y_pred = model.predict(X_test)
    class_report = classification_report(Y_test, y_pred, target_names=category_names)
    print(class_report)


def save_model(model, model_filepath):
    """Saves the model to a pickle file"""
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model t o as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
