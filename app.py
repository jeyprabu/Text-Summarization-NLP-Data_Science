from flask import Flask, render_template, request
from summarizer import text_summarize, sentiment_analysis, word_cloud  
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', result='')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    action = request.form['action']
    answer=None
    if action == 'summarize':
        answer = 'Summarized Text'
        result = text_summarize(text)

    elif action == 'sentiment':
        answer = 'Sentiment Analysis'
        result = sentiment_analysis(text)
        
    elif action == 'wordcloud':
        answer = 'Word Cloud'
        result = word_cloud(text)

    return render_template('result.html', display = answer, result=result)

if __name__ == '__main__':
    app.run(debug=True)
