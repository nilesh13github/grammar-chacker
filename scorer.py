import pandas as pd
import language_tool_python
import re
import nltk

nltk.download('words')
from nltk.corpus import words

# Load English vocabulary
english_vocab = set(words.words())

def is_english_word(word):
    return word.lower() in english_vocab

# Load dataset
df = pd.read_csv("train_data.csv")

# Function to estimate clarity based on English words
def estimate_language_clarity(text):
    word_count = 0
    english_word_count = 0

    # Split and clean the words
    chunks = text.split()
    for w in chunks:
        clean_word = re.sub(r'\W+', '', w).strip().lower()  # Remove punctuation
        if clean_word:  # Skip empty strings
            word_count += 1
            if is_english_word(clean_word):
                english_word_count += 1

    if word_count == 0:
        return 0.0  # Avoid division by zero

    clarity_score = english_word_count / word_count
    return round(clarity_score, 2)

# Apply to your dataframe
df['clarity_score'] = estimate_language_clarity(str(df['text']))

df.to_csv("clearity_scores.csv", index=False)