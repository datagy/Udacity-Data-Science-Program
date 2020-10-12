import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
import re
import pickle
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    """Takes the database_filepath as an input and loads the data. 
    
    Returns:
    X - message data
    y - target values
    column_names - list of column names."""
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages', engine)
    X = df.message
    y = df.drop(['id', 'message', 'genre', 'original'], axis=1)
#     for col in y.columns:
#         y[col] = pd.to_numeric(y[col])
    print(y)
    category_names = y.columns.tolist()
    
    return X, y, category_names


def tokenize(text):
    """Takes text as an input and returns tokenized text."""    
    
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
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    parameters = {
            'vect__ngram_range': ((1, 1), (1, 2)),
            'vect__max_df': (0.5, 0.75, 1.0),
            'tfidf__use_idf': (True, False)
        }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the model. Takes a model, the X_test matrix, Y_test, and category names as inputs. Prints out an evaluation dataframe."""
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=category_names)
    evaluation = {}
    for column in Y_test.columns:
        evaluation[column] = []
        evaluation[column].append(precision_score(Y_test[column], y_pred_df[column]))
        evaluation[column].append(recall_score(Y_test[column], y_pred_df[column]))
        evaluation[column].append(f1_score(Y_test[column], y_pred_df[column]))
    print(pd.DataFrame(evaluation))


def save_model(model, model_filepath):
    """Saves the model to a pickel file."""
    pickle.dump(pipeline, open('model.pkl', 'wb'))


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
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
