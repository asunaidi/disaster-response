# Disaster Response Application

This project is part of Udacity Data Scientist Nanodegree. It builds an app that can be used by the emergency employees to help them promptly assist people in case of disasters. When a disaster happenes, the emergency recieves thousands of messages and it’s hard to deal with all of them at once. This app can read a message sent by someone seeking help and decide which entity is suitable to assist the sender. It uses data provided by Figure Eight that contains previous messages from people seeking help along with their associated categories. Those data will be first cleaned in this project and then used to build a Random forest model that is able to classify a new message into a suitable category.

# Installation
No libraries are needed other than the ones provided by Anaconda. The code was built on Python 3 and HTML.

## File Descriptions
* **data**: This folder contains sample messages and categories datasets in csv format.
* **app**: This folder contains all the files necessary to run the app.
* **/data/process_data.py**: This file takes csv files containing messages and categories as input, and return an SQLite database containing a merged and cleaned version of these two datasets.
* **/model/train_classifier.py**: This file takes an SQLite database as an input and uses the data to build a tuned ML model to classify messages. It returns the fitted model as Pickle file.
* **run.py**: This python file containing all code necessary to deploy the model and visuals some statistics about the data
* **ETL Pipeline Preparation.ipynb**: This Jupyter Notebook file was used to create process_data.py. 
* **ML Pipeline Preparation.ipynb**: This Jupyter Notebook was used to create train_classifier.py.


## Running Instructions
### ***Run process_data.py***
In your terminal, navigate to ‘data folder and run the following command: 
`python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
### ***Run train_classifier.py***
In your terminal, navigate to the ‘models’ folder and run the following command: 
`python train_classifier.py data/DisasterResponse.db classifier.pkl`
### ***Run the Disaster App***
In your terminal, navigate to the ‘app’ folder and run the following command: 
    `python run.py`
Go to http://0.0.0.0:3001/
