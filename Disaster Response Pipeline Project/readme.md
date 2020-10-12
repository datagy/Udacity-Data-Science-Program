# Disaster Response Pipeline Project
Using data provided from Figure Eight, this project's objective is to analyze data from disasters and build a model that classifies these messages. Then, a machine learning pipeline is created to categorize these events as well as future events.

## Instructions to run the code
Run the following commands in the project's root directory to set up your database and model.

1. ETL pipeline - cleans data and stores in database, run: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. ML pipeline - trains classifier and saves model, run: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
3. To run the web app, run: `python run.py` and go to http://0.0.0.0:3001/ in your browswer
