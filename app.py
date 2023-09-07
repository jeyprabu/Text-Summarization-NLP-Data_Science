import os
import spacy
from flask import Flask, render_template, request, redirect, url_for
from summarizer import generate_summary 
import requests
from textblob import TextBlob

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def perform_sentiment_analysis(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form.get('input_text')
        method = request.form.get('method')
        ratio = float(request.form.get('ratio'))
        
        try:
            summary = generate_summary(input_text, method=method, ratio=ratio)
            sentiment = perform_sentiment_analysis(input_text)
            entities = extract_entities(input_text) 
            return render_template('result.html', summary=summary, sentiment=sentiment, entities=entities)
        except Exception as e:
            error_message = str(e)
            return render_template('index.html', error_message=error_message)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
