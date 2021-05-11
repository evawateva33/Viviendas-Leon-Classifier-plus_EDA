import pickle 
import pandas as pd  #Pandas for data pre-processing
import joblib
import pickle #Pickle for pickling (saving) the model 

# # some time later...
import flask
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
 #app name
import pickle
# Use pickle to load in the pre-trained model.
with open(f'model/farm2_model_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)


app = flask.Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        Disease = flask.request.form['% Disease']
        Wellness_Condition = flask.request.form['Wellness_Condition']
        HeatIndexC = flask.request.form['HeatIndexC']
        DewPointC = flask.request.form['DewPointC']
        WindChillC = flask.request.form['WindChillC']
        Season_Fall = flask.request.form['Season_Fall']
        sunHour = flask.request.form['sunHour']
        Season_Spring = flask.request.form['Season_Spring']
        Season_Summer = flask.request.form['Season_Summer']
        Season_Winter = flask.request.form['Season_Winter']
        Region_Goyena = flask.request.form['Region_Goyena']
        Region_Troilo = flask.request.form['Region_Troilo']
        
        input_variables = pd.DataFrame([[Disease, Wellness_Condition, HeatIndexC,
                                        DewPointC,WindChillC,sunHour,Season_Fall,Season_Spring
                                         ,Season_Summer,Season_Winter,Region_Goyena,Region_Troilo]],
                                       columns=['% Disease', 'Wellness_Condition', 'HeatIndexC','DewPointC', 
                                                'WindChillC', 'sunHour', 'Season_Fall', 'Season_Spring', 
                                                'Season_Summer', 'Season_Winter', 'Region_Goyena', 
                                                'Region_Troilo'],
                                       dtype=float)
        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html',
                                     original_input={'% Disease': Disease,
                                                     'Wellness_Condition':Wellness_Condition,
                                                     'Region_Troilo':Region_Troilo,
                                                     'HeatIndexC':HeatIndexC,
                                                    'DewPointC':DewPointC,
                                                     'WindChillC':WindChillC,
                                                     'sunHour':sunHour,
                                                     'Season_Fall':Season_Fall,
                                                     'Season_Spring':Season_Spring,
                                                    'Season_Summer':Season_Summer,
                                                    'Season_Winter':Season_Winter,
                                                    'Region_Goyena':Region_Goyena },
                                     result=prediction,
                                     )
if __name__ == '__main__':
    app.run()
