import sys

# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import pickle 
from sklearn.pipeline import Pipeline



def load_data(database_filepath):
    """
    Load data from database
    
    Parameters:
        database_filepath: the filepath of database wich data stored.
    Return
        X: feature variables.
        Y: target variables.
        category_names: category name list
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('CleanMessage', engine)
    
    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis=1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    """
    Process your text data with a tokenization way
    
    Parameters:
        text: text need to be tokenized.
    Return
        clean_tokens: word list.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline.
    
    Parameters:
        na
    Return
        cv: GridSearchCV with multi output classifier.
    """
    pipeline = Pipeline([
         ('vect', CountVectorizer(tokenizer=tokenize)),
         ('tfidf', TfidfTransformer()),
         ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
     ])
    parameters = {
        'clf__n_jobs':[None, 10]
     }

    cv = GridSearchCV(pipeline, parameters)
    
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate a machine learning argorithm.
    
    Parameters:
        model: a model
        X_test: training target variables.
        Y_test: testing target varirables.
        categories_names: catgory name list
    Return
        na
    """
    y_pred = model.predict(X_test)
    i=0

    for column in category_names:
         print('{}\n'.format,column)
         print(classification_report(Y_test[column], y_pred[:,i]))
         i += 1


def save_model(model, model_filepath):
    """
    save a machine learning model.
    
    Parameters:
        model: a model
        model_filepath: a file path need to be saved.
    Return
        na
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

    f.close()


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
