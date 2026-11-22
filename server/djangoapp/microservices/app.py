
from flask import Flask
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import json
import os

app = Flask("Sentiment Analyzer")

# Tell NLTK where to find the sentiment lexicon
nltk.data.path.append('/python-docker/sentiment')

# Extract the VADER lexicon if it is not already installed
lexicon_zip = '/python-docker/sentiment/vader_lexicon.zip'
if os.path.exists(lexicon_zip):
    nltk.download(
        'vader_lexicon',
        download_dir='/python-docker/sentiment',
        quiet=True
    )

sia = SentimentIntensityAnalyzer()


@app.get('/')
def home():
    return "Welcome to the Sentiment Analyzer. Use /analyze/text to get the sentiment"


@app.get('/analyze/<input_txt>')
def analyze_sentiment(input_txt):
    scores = sia.polarity_scores(input_txt)

    pos = float(scores['pos'])
    neg = float(scores['neg'])
    neu = float(scores['neu'])

    res = "positive"

    if neg > pos and neg > neu:
        res = "negative"
    elif neu > neg and neu > pos:
        res = "neutral"

    return json.dumps({"sentiment": res})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
