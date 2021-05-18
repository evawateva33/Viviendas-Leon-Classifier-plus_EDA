import pickle 
import pandas as pd  #Pandas for data pre-processing
import joblib
import pickle #Pickle for pickling (saving) the model 
import xgboost
# # some time later...
import flask
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
 #app name
import pickle
import collections
from collections import defaultdict

# Use pickle to load in the pre-trained model.
with open(f'model/farm33_model_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)
with open(f'model/farm2_model_xgboost.pkl', 'rb') as f:
    model2 = pickle.load(f)

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
        df = pd.read_csv('testData/vl_geow_f.csv')
   
        # 1) Remove the low crops
         # Getting counts of crops in dataframe
        crop_counts = df.groupby(['Crop']).size().sort_values(ascending=True)
          # Selecting index crop names with less than 10 counts
        low_crops = crop_counts[crop_counts < 10].index.tolist()
        # filtering dataframe without
        df_less = df[~df['Crop'].isin(low_crops)]
        # 2) Train the model
        predictorsc = df_less[['% Disease', 'Wellness_Condition', 'HeatIndexC', 'DewPointC', 'WindChillC', 'sunHour', 'Season_Fall', 'Season_Spring', 'Season_Summer', 'Season_Winter', 'Region_Goyena', 'Region_Troilo']]
        targetc = df_less['Crop']
        input_variables = pd.DataFrame([[Disease, Wellness_Condition, HeatIndexC,
                                        DewPointC,WindChillC,sunHour,Season_Fall,Season_Spring
                                         ,Season_Summer,Season_Winter,Region_Goyena,Region_Troilo]],
                                       columns=['% Disease', 'Wellness_Condition', 'HeatIndexC','DewPointC', 
                                                'WindChillC', 'sunHour', 'Season_Fall', 'Season_Spring', 
                                                'Season_Summer', 'Season_Winter', 'Region_Goyena', 
                                                'Region_Troilo'],
                                       dtype=float)
        crop_model = model.predict(input_variables)[0]
        some_conditions = [0, 100, 30.74, 20.66, 28.10, 10.95, 0, 0, 0, 0, 0, 0]
        columns_dict = {0: '% Disease', 1: 'Wellness_Condition', 2: 'HeatIndexC',
                  3: 'DewPointC', 4: 'WindChillC', 5: 'sunHour',
                  6: 'Season_Fall', 7: 'Season_Spring', 8: 'Season_Summer',
                  9: 'Season_Winter', 10: 'Region_Goyena', 11: 'Region_Troilo'}
        conditions_df = pd.DataFrame(some_conditions).T.rename(columns=columns_dict)


        type_model = model2.predict(input_variables)[0]

        #well_classified_crops = class_assessment(crop_model, predictorsc, targetc)
        crop_scores = defaultdict(list)
        classes = crop_model.classes_
        X_train, X_test, y_train, y_test = train_test_split(predictorsc.values, targetc.values, test_size=0.2, shuffle=True)
        kf = KFold(n_splits=3, random_state=42)
        for train_ind, val_ind in kf.split(X_train, y_train):

      # Split train into validation sets
            X_tr, y_tr = X_train[train_ind], y_train[train_ind]
            X_val, y_val = X_train[val_ind], y_train[val_ind]
            # Get roc auc score for each crop
            for each in classes:
                fpr, tpr, thresholds = roc_curve(y_val,  
                    crop_model.fit(X_tr, y_tr).predict_proba(X_val)[:,1], pos_label = each)
                auc = round(metrics.auc(fpr, tpr),2)
                crop_scores[each].append(auc)

            crop_auc = pd.DataFrame.from_dict(crop_scores, orient='index')
            crop_auc['avg'] = crop_auc.mean(axis=1)
        
        crop_auc2 = crop_auc[crop_auc['avg'] > 0.5]
        crop_auc2.drop(crop_auc.columns[[0, 1, 2]], axis=1, inplace=True)
        crop_auc2.sort_values(by=['avg'], ascending=False, inplace=True)
        well_classified_crops = crop_auc2


        
        well_classified_crops.reset_index().rename(columns={'index': 'Crop'})
        well_classified_crops = well_classified_crops.head(3)
    

          # 5) Call the scoring system
        score = init_score(df)
        result = score()
        if Region_Goyena == 1:
            region = 'Goyena'
        elif Region_Troilo == 1:
            region = 'Troilo'
        else:
            region = None
        score_func = init_score(df)
        result = score_func(region)
        high_score_crops = result.get_best_composite(n=3)
        high_score_categories = result.get_best_type_composite(n=1)
        if high_score_categories[0] == 'Veg':
            high_score_categories[0] = 'Vegetable'

      # 6) Add the results of the crop scoring to get the crop DF up to 3
        crops_length = len(well_classified_crops)
        if crops_length < 3:
            high_score_crops_2D = []
        for i in range(min(len(high_score_crops, 3 - crops_length))):
            high_score_crops_2D.append([high_score_crops[i], None])
            high_score_df = pd.DataFrame(new_high_score_crops, columns=['Crop', 'avg'])
        well_classified_crops = well_classified_crops.append(high_score_df)
  
 
  
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
                                     result=[well_classified_crops, type_model],
                                     )
if __name__ == '__main__':
    app.run()
