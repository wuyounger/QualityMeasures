import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
import tweepy
import time
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
import spacy
import pytextrank
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import json
import re
from flashtext import KeywordProcessor


# --------------------------------------
# BASIC APP SETUP
# --------------------------------------
app = Flask(__name__, instance_relative_config=True)
# english = spacy.load("en_core_web_sm")

# Config
app_settings = os.getenv(
    'APP_SETTINGS',
    'config.DevelopmentConfig'
)
app.config.from_object(app_settings)
  

# Extensions
from flask_cors import CORS
CORS(app)

#SENTIMENT CONFIG
SEQUENCE_LENGTH = 300
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)


# Twitter
auth = tweepy.OAuthHandler(app.config.get('CONSUMER_KEY'), app.config.get('CONSUMER_SECRET'))
auth.set_access_token(app.config.get('ACCESS_TOKEN'), app.config.get('ACCESS_TOKEN_SECRET'))
api = tweepy.API(auth,wait_on_rate_limit=True)


from tensorflow import keras 
model = keras.models.load_model('./model.h5')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt

def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE


def predict(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}

import requests

key_word_list = ['fix', 'broken', 'unable','ticket','uninstalled','disabled','difficulty','hard','annoying','ban','shit','fuck','bug']
keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_list(key_word_list)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyzehashtag', methods=['GET'])
def analyzehashtag():
    positive = 0
    neutral = 0
    negative = 0

    search_results =  [] 
    query = request.args.get("query")
    print(query)
    for tweet in tweepy.Cursor(api.search, q = str(query) + '-filter:retweets' ,rpp=5,lang="en", tweet_mode='extended').items(500):
        
        clean_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", tweet.full_text).split())
        result = predict(clean_text)

        if(result["label"] == POSITIVE):
            positive += 1
        if(result["label"] == NEUTRAL):
            neutral += 1
        if(result["label"] == NEGATIVE):
            negative += 1
            keywords_found = keyword_processor.extract_keywords(clean_text, max_cost=3)
            if len(keywords_found) != 0:
                url = 'https://twitter.com/twitter/statuses/' + str(tweet.id)
                text = tweet.full_text
                tw = {'url':url, 'text':text, 'canonical_link': url}
                search_results.append(tw)

    if not search_results:
        return no_results_template(query)
    return render_template('search_results.html', search_results=search_results, query=query)        

def no_results_template(query):
    return render_template('simple_message.html', title='No results found',
                           message='Your search - <b>' + query + '</b> - did not match any documents.'
                                                                 '<br>Suggestions:<br><ul>'
                                                                 '<li>Make sure that all words are spelled correctly.</li>'
                                                                 '<li>Try different keywords.</li>'
                                                                 '<li>Try more general keywords.</li>'
                                                                 '<li>Try fewer keywords.</ul>')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)