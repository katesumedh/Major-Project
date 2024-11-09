import pandas as pd
from flask import Flask, request, render_template, jsonify
from transformers import pipeline
import random
import os

# Initialize the sentiment-analysis pipeline
sent_pipeline = pipeline("sentiment-analysis")

app = Flask(__name__)

# Function to generate Credit Score based on sentiment
def generate_credit_score(sentiment):
    if sentiment == 'NEGATIVE':
        return random.randint(350, 650)
    else:  # Positive sentiment
        return random.randint(650, 890)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_csv', methods=['POST'])
def analyze_csv():
    # Get the file from the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file and file.filename.endswith('.csv'):
        # Read the CSV file
        df = pd.read_csv(file)
        
        results = []

        # Process each row in the CSV file
        for index, row in df.iterrows():
            text = row['Text']
            sentiment_result = sent_pipeline(text)[0]
            sentiment = sentiment_result['label']
            score = sentiment_result['score']
            
            # Generate a random credit score based on sentiment
            credit_score = generate_credit_score(sentiment)
            
            # Prepare the result
            results.append({
                'Activity': sentiment, 
                'Confidence': score,
                'Credit Score': credit_score,
                'Text': text
            })

        # Convert results into DataFrame
        result_df = pd.DataFrame(results)
        
        # Save the result to a new CSV file
        result_filename = os.path.join('static', 'sentiment_results.csv')
        result_df.to_csv(result_filename, index=False)

        return jsonify({"message": "File processed successfully", "filename": result_filename})

    return jsonify({"error": "Invalid file format. Please upload a CSV file."})

if __name__ == '__main__':
    app.run(debug=True)
