import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    #data for first graph
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    #data for second graph
    categories = df.drop(['id', 'message', 'original', 'genre'], axis=1).astype(float)
    categories_names = categories.columns.values
    categories_counts = [sum(categories[cat]) for cat in categories_names]
    #data for the third graph
    correlations = categories.corr().values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'How Were the Messages Recieved?',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_counts
                )
            ],

            'layout': {
                'title': 'Number of Messages in Each Category',
                'yaxis': {
                    'title': "Count"
                },
                'height' : 800,
                'margin': dict(
                    b = 230,
                ),
            }
        },
        
        {
            'data': [
                 Heatmap(
                    z=correlations,
                    x=categories_names,
                    y=categories_names
                )
            ],

            'layout': {
                'title': 'How Categories of Messages are Correlated?',
                'height': 800,
                'margin': dict(
                    l = 150,
                    r = 20, 
                    b = 150,
                    t = 50
                 ),
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()