{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app \"__main__\" (lazy loading)\n",
      " * Environment: production\n",
      "\u001b[31m   WARNING: This is a development server. Do not use it in a production deployment.\u001b[0m\n",
      "\u001b[2m   Use a production WSGI server instead.\u001b[0m\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Running on http://127.0.0.1:7525/ (Press CTRL+C to quit)\n",
      "127.0.0.1 - - [10/May/2021 09:33:00] \"\u001b[37mGET / HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [10/May/2021 09:33:00] \"\u001b[33mGET /paper.gif HTTP/1.1\u001b[0m\" 404 -\n",
      "127.0.0.1 - - [10/May/2021 09:33:00] \"\u001b[33mGET /favicon.ico HTTP/1.1\u001b[0m\" 404 -\n"
     ]
    }
   ],
   "source": [
    "import pickle \n",
    "import pandas as pd  #Pandas for data pre-processing\n",
    "import joblib\n",
    "import pickle #Pickle for pickling (saving) the model \n",
    "\n",
    "from sklearn.feature_extraction.text import CountVectorizer #To Vectorize the textual data \n",
    "\n",
    "from sklearn.naive_bayes import MultinomialNB #The algorithm for prediction \n",
    "\n",
    " #Alternative Usage of Saved Model \n",
    "\n",
    "from sklearn.model_selection import train_test_split #Validation split\n",
    "# filename = 'finalized_model2.sav'\n",
    "# pickle.dump(SVC, open(filename, 'wb'))\n",
    " \n",
    "# # some time later...\n",
    " \n",
    "\n",
    "import pickle\n",
    "# Use pickle to load in the pre-trained model.\n",
    "with open(f'model/farm2_model_xgboost.pkl', 'rb') as f:\n",
    "    model = pickle.load(f)\n",
    "\n",
    "import flask\n",
    "app = flask.Flask(__name__, template_folder='templates')\n",
    "@app.route('/', methods=['GET', 'POST'])\n",
    "def main():\n",
    "    if flask.request.method == 'GET':\n",
    "        return(flask.render_template('main.html'))\n",
    "    if flask.request.method == 'POST':\n",
    "        Disease = flask.request.form['% Disease']\n",
    "        Wellness_Condition = flask.request.form['Wellness_Condition']\n",
    "        HeatIndexC = flask.request.form['HeatIndexC']\n",
    "        DewPointC = flask.request.form['DewPointC']\n",
    "        WindChillC = flask.request.form['WindChillC']\n",
    "        Season_Fall = flask.request.form['Season_Fall']\n",
    "        sunHour = flask.request.form['sunHour']\n",
    "        Season_Spring = flask.request.form['Season_Spring']\n",
    "        Season_Summer = flask.request.form['Season_Summer']\n",
    "        Season_Winter = flask.request.form['Season_Winter']\n",
    "        Region_Goyena = flask.request.form['Region_Goyena']\n",
    "        Region_Troilo = flask.request.form['Region_Troilo']\n",
    "        \n",
    "        input_variables = pd.DataFrame([[Disease, Wellness_Condition, HeatIndexC,\n",
    "                                        DewPointC,WindChillC,sunHour,Season_Fall,Season_Spring\n",
    "                                         ,Season_Summer,Season_Winter,Region_Goyena,Region_Troilo]],\n",
    "                                       columns=['% Disease', 'Wellness_Condition', 'HeatIndexC','DewPointC', \n",
    "                                                'WindChillC', 'sunHour', 'Season_Fall', 'Season_Spring', \n",
    "                                                'Season_Summer', 'Season_Winter', 'Region_Goyena', \n",
    "                                                'Region_Troilo'],\n",
    "                                       dtype=float)\n",
    "        prediction = model.predict(input_variables)[0]\n",
    "        return flask.render_template('main.html',\n",
    "                                     original_input={'% Disease': Disease,\n",
    "                                                     'Wellness_Condition':Wellness_Condition,\n",
    "                                                     'Region_Troilo':Region_Troilo,\n",
    "                                                     'HeatIndexC':HeatIndexC,\n",
    "                                                    'DewPointC':DewPointC,\n",
    "                                                     'WindChillC':WindChillC,\n",
    "                                                     'sunHour':sunHour,\n",
    "                                                     'Season_Fall':Season_Fall,\n",
    "                                                     'Season_Spring':Season_Spring,\n",
    "                                                    'Season_Summer':Season_Summer,\n",
    "                                                    'Season_Winter':Season_Winter,\n",
    "                                                    'Region_Goyena':Region_Goyena },\n",
    "                                     result=prediction,\n",
    "                                     )\n",
    "if __name__ == '__main__':\n",
    "    app.run(host='127.0.0.1',port=7525,debug=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
