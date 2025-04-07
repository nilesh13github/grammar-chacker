import pandas as pd
import pickle
import scipy.sparse

# Load the saved model
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the saved TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load new data from CSV (ensure it contains 'text' and 'clarity_score' columns)
new_data = pd.read_csv("test_scores.csv")

# Extract text and clarity_score from the new data
new_text = new_data['text'].tolist()
new_clarity = new_data['clarity_score'].values.reshape(-1, 1)

# Vectorize the text using the loaded TF-IDF vectorizer
new_text_vec = vectorizer.transform(new_text)

# Convert the clarity scores to a sparse matrix
new_clarity_sparse = scipy.sparse.csr_matrix(new_clarity)

# Combine the TF-IDF features with the clarity_score feature
new_combined_features = scipy.sparse.hstack([new_text_vec, new_clarity_sparse])

# Use the model to predict the label
predicted_label = model.predict(new_combined_features)

# Since your training label was defined as labels/10,
# multiply the predicted label by 10 to recover the original scale of the clarity score.
predicted_label = predicted_label * 10

# Optionally add predictions to the dataframe and display or save them
# Add predictions to the dataframe
new_data['predicted_clarity_score'] = predicted_label

# Round off the clarity_score and predicted_clarity_score to 2 decimal places
new_data['clarity_score'] = new_data['clarity_score'].round(2)
new_data['predicted_clarity_score'] = new_data['predicted_clarity_score'].apply(lambda x: round(x * 2) / 2) - 0.5

# Optionally, print the results
print(new_data[['filename', 'predicted_clarity_score']])


new_data[['filename', 'predicted_clarity_score']].to_csv("predictions.csv", index=False)