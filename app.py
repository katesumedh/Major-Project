from flask import Flask, render_template, request
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import random

app = Flask(__name__)

# Load the model (make sure to use your own pickle file)
MODEL_PATH = "roberta_model.pkl"  # Path to your pickle file
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Function to get sentiment using RoBERTa
def get_sentiment(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for processing sentiment analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        # Get the input text from the form
        input_text = request.form['text']
        
        # Get sentiment results
        sentiment = get_sentiment(input_text)
        
        # Find the dominant sentiment
        if sentiment['roberta_pos'] > sentiment['roberta_neg'] and sentiment['roberta_pos'] > sentiment['roberta_neu']:
            label = 'POSITIVE'
            score = sentiment['roberta_pos']
            credit_score = random.randint(700, 900)
        elif sentiment['roberta_neg'] > sentiment['roberta_pos'] and sentiment['roberta_neg'] > sentiment['roberta_neu']:
            label = 'NEGATIVE'
            score = sentiment['roberta_neg']
            credit_score = random.randint(350, 600)
        else:
            label = 'NEUTRAL'
            score = sentiment['roberta_neu']
            credit_score = random.randint(600, 700)
        
        return render_template('index.html', label=label, score=score, input_text=input_text, credit_score = credit_score)

if __name__ == "__main__":
    app.run(debug=True)
