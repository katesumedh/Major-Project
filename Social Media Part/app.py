from flask import Flask, render_template, request
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import random
import os

app = Flask(__name__)

# Load the pretrained model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Function to get sentiment using RoBERTa
def get_sentiment(text):
    # Split the text into chunks of 512 tokens (or smaller if needed)
    chunk_size = 512  # You can adjust this based on the model's max token length
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    
    # Initialize score variables
    total_scores = {'roberta_neg': 0, 'roberta_neu': 0, 'roberta_pos': 0}
    
    for chunk in chunks:
        # Convert chunk back to text
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        
        # Tokenize the chunk with truncation and padding
        encoded_text = tokenizer(chunk_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
        # Get the model's output
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        
        # Apply softmax to the scores
        scores = softmax(scores)
        
        # Aggregate the scores
        total_scores['roberta_neg'] += scores[0]
        total_scores['roberta_neu'] += scores[1]
        total_scores['roberta_pos'] += scores[2]
    
    # Normalize the scores to get the final sentiment
    total_score_sum = sum(total_scores.values())
    normalized_scores = {key: score / total_score_sum for key, score in total_scores.items()}
    
    return normalized_scores


# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for processing sentiment analysis (for CSV)
@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return 'No file part', 400
        
        file = request.files['file']
        
        if file.filename == '':
            return 'No selected file', 400
        
        # Get the value of the dropdown
        dropdown_value = int(request.form.get('dropdown'))

        # Ensure the file is a CSV
        if file and file.filename.endswith('.csv'):
            # Read the CSV into a pandas dataframe
            df = pd.read_csv(file)

            # Combine all the rows in the CSV as one large block of text
            combined_text = ' '.join(df.astype(str).apply(' '.join, axis=1))

            # Get sentiment for the combined text
            sentiment = get_sentiment(combined_text)
            
            # Find the dominant sentiment
            if sentiment['roberta_pos'] > sentiment['roberta_neg'] and sentiment['roberta_pos'] > sentiment['roberta_neu']:
                label = 'POSITIVE'
                score = sentiment['roberta_pos']
                if dropdown_value == 0:
                    credit_score = random.randint(550, 700)
                elif dropdown_value == 1:
                    credit_score = random.randint(700, 800)
                else:
                    credit_score = random.randint(800, 890)
                
            elif sentiment['roberta_neg'] > sentiment['roberta_pos'] and sentiment['roberta_neg'] > sentiment['roberta_neu']:
                label = 'NEGATIVE'
                score = sentiment['roberta_neg']
                if dropdown_value == 0:
                    credit_score = random.randint(300, 400)
                elif dropdown_value == 1:
                    credit_score = random.randint(400, 500)
                else:
                    credit_score = random.randint(500, 600)

            else:
                label = 'NEUTRAL'
                score = sentiment['roberta_neu']
                if dropdown_value == 0:
                    credit_score = random.randint(550, 650)
                elif dropdown_value == 1:
                    credit_score = random.randint(650, 700)
                else:
                    credit_score = random.randint(700, 750)

            # Return the result to the template with the dropdown value
            result = {
                'label': label,
                'score': score,
                'credit_score': credit_score, # Add the dropdown value to the result
            }

            return render_template('index.html', result=result)

        else:
            return 'Invalid file format. Please upload a CSV file.', 400

if __name__ == "__main__":
    app.run(debug=True)
