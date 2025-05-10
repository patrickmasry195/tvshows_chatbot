# Import necessary libraries
from flask import Flask, request, jsonify
from flask_cors import CORS  # For handling CORS issues
import pandas as pd
import numpy as np
from transformers import pipeline
from spellchecker import SpellChecker
from rapidfuzz import process, fuzz
from sklearn.metrics.pairwise import cosine_similarity
import re
import random

app = Flask(__name__)
CORS(app)

# Initialize your existing components
df = pd.read_csv("final_series_profile.csv")
embeddings_df = pd.read_csv("final_series_embeddings.csv")
EXCLUDED_TITLES = {"movie movie"}
STOP_WORDS = {'a', 'an', 'the', 'and', 'or', 'like', 'i', 'want'}

# Initialize Spell Checker
spell = SpellChecker(language='en', case_sensitive=False)
custom_movie_titles = {title for title in df["title"]}
spell.word_frequency.load_words(custom_movie_titles)

# Create lowercase to original title mapping
title_lower_to_original = {title.lower(): title for title in df["title"]}
MOVIE_DATABASE = list(title_lower_to_original.keys())

# Load NER Model
ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    tokenizer="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

# Your existing INTENTS dictionary
INTENTS = {
    "greeting": {
        "patterns": [r"hi|hello|hey|good morning|good evening|how are you|how are you doing"],
        "responses": ["Hello! How can I help you today?", "Hi there! Ready for series recommendations?","Hey, hope you are doing well", "Hey, are you ready to watch new series today"]
    },
    "leaving": {
        "patterns": [r"bye|goodbye|exit|quit|see you"],
        "responses": ["Goodbye! Happy watching!", "See you later!"]
    },
    "services": {
        "patterns": [r"what can you do|service|help"],
        "responses": ["I can recommend series! Just ask for recommendations and mention a series you like.","I am here to help you finding new series"]
    },
    "recommendation": {
        "patterns": [r"recommend|suggest|series like|a series like"],
    },
    "non_related": {
        "responses": ["Hmm, I'm not sure about that. Let's stick to series recommendations!","I am not programmed to answer these questions"]
    }
}

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def detect_intent(user_input):
    cleaned_input = preprocess(user_input)
    for intent, data in INTENTS.items():
        if intent == "non_related":
            continue  # Handle last
        if "patterns" in data:
            for pattern in data["patterns"]:
                if re.search(pattern, cleaned_input):
                    return intent
    return "non_related"  # Default if no match

def correct_spelling(query):
    """Correct spelling while preserving title case."""
    words = query.split()
    corrected_words = []
    for word in words:
        if word.lower() in [t.lower() for t in custom_movie_titles]:
            original_title = next(t for t in custom_movie_titles if t.lower() == word.lower())
            corrected_words.append(original_title)
        else:
            corrected_word = spell.correction(word.lower())
            if word.istitle():
                corrected_word = corrected_word.title()
            elif word.isupper():
                corrected_word = corrected_word.upper()
            corrected_words.append(corrected_word)
    return ' '.join(corrected_words)

def generate_ngrams(words, max_n=5):
    """Generate n-grams from tokenized words."""
    ngrams = []
    for n in range(1, max_n+1):
        for i in range(len(words)-n+1):
            ngrams.append(' '.join(words[i:i+n]).lower())
    return ngrams

def extract_movie_titles(query):
    # Step 1: Correct Spelling
    corrected_query = correct_spelling(query)
    
    # Step 2a: Extract NER Candidates (multi-word supported)
    results = ner_pipeline(corrected_query)
    ner_candidates = [
        entity["word"].lower() 
        for entity in results 
        if entity["entity_group"] in ["ORG", "MISC"]
    ]
    
    # Step 2b: Generate N-gram Candidates
    corrected_words = corrected_query.split()
    ngram_candidates = generate_ngrams(corrected_words)
    
    # Combine and deduplicate candidates
    all_candidates = list(set(ner_candidates + ngram_candidates))
    
    # Step 3: Context-Aware Filtering
    filtered_candidates = []
    for candidate in all_candidates:
        if ' ' in candidate:
            filtered_candidates.append(candidate)
        else:
            if len(candidate) > 2 and candidate not in STOP_WORDS:
                filtered_candidates.append(candidate)
    
    # Step 4: Validate with Fuzzy Matching
    validated_titles = []
    for candidate in sorted(filtered_candidates, key=lambda x: -len(x)):
        match, score, _ = process.extractOne(
            candidate,
            MOVIE_DATABASE,
            scorer=fuzz.WRatio
        )
        if score > 90 and match not in EXCLUDED_TITLES:
            validated_titles.append(title_lower_to_original[match])
            break
    
    return validated_titles[0] if validated_titles else ""

def find_similar_movies(input_title):
    """Find top 5 similar series based on embeddings."""
    try:
        # Get embeddings
        embedding_columns = embeddings_df.columns.difference(['title'])
        embeddings = embeddings_df[embedding_columns].values
        
        # Locate the index of the input title
        index = embeddings_df[embeddings_df['title'] == input_title].index[0]
        input_embedding = embeddings[index]
        
        # Compute cosine similarity
        similarities = cosine_similarity(input_embedding.reshape(1, -1), embeddings).flatten()
        
        # Get top 5 similar movies (excluding the input itself)
        top_indices = similarities.argsort()[::-1][1:6]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append(embeddings_df.iloc[idx]['title'])
        
        return recommendations
    except Exception as e:
        return None

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.json
        user_message = data.get('message')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Process the message using your existing logic
        corrected_input = correct_spelling(user_message)
        cleaned_input = preprocess(corrected_input)
        intent = detect_intent(cleaned_input)
        
        if intent == "greeting":
            response = {
                "response": random.choice(INTENTS[intent]["responses"])
            }
        elif intent == "leaving":
            response = {
                "response": random.choice(INTENTS[intent]["responses"])
            }
        elif intent == "services":
            response = {
                "response": random.choice(INTENTS[intent]["responses"])
            }
        elif intent == "recommendation":
            movie_title = extract_movie_titles(corrected_input)
            
            if not movie_title:
                response = {
                    "response": "Could not identify a series title in your request."
                }
            else:
                recommendations = find_similar_movies(movie_title)
                if recommendations:
                    response = {
                        "response": {
                            "movie": movie_title,
                            "recommendations": recommendations
                        }
                    }
                else:
                    response = {
                        "response": "Sorry, I couldn't find any recommendations for that series."
                    }
        else:  # non_related
            response = {
                "response": random.choice(INTENTS["non_related"]["responses"])
            }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)