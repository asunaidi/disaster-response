import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
import pickle


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    
    
    """
    This function is used to load the clean messages data and split them into input and target datasets
    Args:

    database_filename: string. the path of the SQLite database containing cleaned message data.
    
    Returns:

    X: dataframe. Dataframe containing features dataset.

    Y: dataframe. Dataframe containing target dataset.

    category_names: list of strings. List containing category names.

    """
    
    # load the data
    full_path = 'sqlite:///' + database_filepath
    engine = create_engine(full_path)
    database_name = database_filepath[:-3]
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    
    #create input (x) and target (y) datasets
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)   
    
    #create a list of category names
    category_names = Y.columns.values
    
    return X, Y, category_names
    


def tokenize(text):
    
    """
    This function is used to normalize, tokenize and stem text string

    Args:

    text: string. String containing message for processing

      
    Returns:

    stemmed: list of strings. List containing normalized and stemmed word tokens

    """
    
     # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    
    """This fnuction is used to build a ML pipeline
    
    Args: None

    Returns:

    cv: gridsearchcv object. Gridsearchcv object that finds the optimal model parameters.

    """
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    
    parameters = {
        'clf__estimator__max_depth': [10, 20],
        'clf__estimator__min_samples_leaf': [1, 2, 4],
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }

    cv = GridSearchCV(pipeline, param_grid = parameters, scoring='f1_micro', n_jobs=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """This function is used to measure accuracy, precision, recall and F1 score for given model
   
    Args:

    model: model object. trained model to classify messages.

    X_test: dataframe. Dataframe containing test features dataset.

    Y_test: dataframe. Dataframe containing test targets dataset.

    category_names: list of strings. List containing category names.


    Returns: None

    """
    Y_pred_tuned = model.best_estimator_.predict(X_test)
    print(classification_report(Y_test.values, Y_pred_tuned, target_names=category_names))


def save_model(model, model_filepath):
    """
    Pickle trained model

    Args:

    model: model object. trained model

    model_filepath: string. path of the file you want to save the model in    

    Returns: None

    """

    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))



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