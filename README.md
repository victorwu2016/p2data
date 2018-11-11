### Table of Contents
1. [Instructions](#instructioins)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Instructions<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Project Motivation<a name="motivation"></a>
The Disaster Response Pipeline Project is to analyze disater data from Figure Eight to build a model for an API that classifies disaster messages.

The files are related to one course ownd by Udacity, it is for practising purpose.

## File Descriptions <a name="files"></a>
The below is the main files:
data/process_data.py: python file to extract, transform data and load them into a database file.
README.md: this file
models/train_classifier.py:  python file to train disaster messages.
models/classifier.pkl:  pickle file of the machine learning argorithm.
app/run.py:  python file to run web app.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Figure Eight for the data.  
