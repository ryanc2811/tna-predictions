from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

app = Flask(__name__)

# Load the model from the file
model = load('model.joblib')

# Load the CSV files into DataFrames
user_profiles = pd.read_csv('./data/user_profiles.csv')
user_responses = pd.read_csv('./data/user_responses.csv')


def create_interaction_matrix(df, user_col, item_col, rating_col, threshold=None):
    """
    :param df: DataFrame containing user, item, and rating columns
    :param user_col: name of the user column
    :param item_col: name of the item/question column
    :param rating_col: name of the rating/response column
    :param threshold: minimum number of ratings required for a user to be included
    :return: interaction matrix as a sparse matrix, user index, item index
    """
    # Filter users below the threshold
    if threshold is not None:
        user_count = df.groupby(user_col)[rating_col].count()
        df = df[df[user_col].isin(user_count[user_count >= threshold].index)]

    # Pivot the DataFrame to create the interaction matrix
    interaction_df = df.pivot(index=user_col, columns=item_col, values=rating_col)
    interaction_df = interaction_df.fillna(0)  # Fill missing values with 0 or your chosen neutral value

    # Create a sparse matrix for more efficient computations
    user_index = interaction_df.index
    item_index = interaction_df.columns
    interaction_matrix = csr_matrix(interaction_df.values)

    return interaction_matrix, user_index, item_index

def predict_scores(interaction_matrix, user_similarity):
    # Ensure interaction_matrix is dense if it's sparse
    if hasattr(interaction_matrix, "toarray"):
        interaction_matrix_dense = interaction_matrix.toarray()
    else:
        interaction_matrix_dense = interaction_matrix

    # Compute predicted scores as a dot product of user similarity and interaction matrix
    predicted_scores = np.dot(user_similarity, interaction_matrix_dense) / np.array([np.abs(user_similarity).sum(axis=1)]).T

    return predicted_scores

def rank_questions(predicted_scores, item_index):
    # Ensure predicted_scores is a dense array if it's sparse
    if hasattr(predicted_scores, "toarray"):
        predicted_scores_dense = predicted_scores.toarray()
    else:
        predicted_scores_dense = predicted_scores

    # Get the indices of the scores sorted in descending order for each user
    ranked_question_indices = np.argsort(-predicted_scores_dense, axis=1)

    # Convert indices to question IDs using the item_index list
    ranked_questions = []
    for user_rank in ranked_question_indices:
        ranked_questions.append([item_index[i] for i in user_rank])

    return ranked_questions

def get_recommendations(new_user_profile_dict):
    
    # Convert the new user profile dictionary to DataFrame
    new_user_profile = pd.DataFrame([new_user_profile_dict])
    cluster = model.predict(new_user_profile)[0]
    
    # Filter users in the same cluster and compute similarities, etc.
    similar_users = user_profiles[user_profiles['cluster'] == cluster]['user_id']

    # Assuming `similar_users` contains user IDs of similar users and `user_responses` is your DataFrame with all user responses
    filtered_responses = user_responses[user_responses['user_id'].isin(similar_users)]

    # Create the user-item interaction matrix
    interaction_matrix, user_index, item_index = create_interaction_matrix(filtered_responses, 'user_id', 'question_id', 'response')

    # Compute user similarity matrix
    user_similarity = cosine_similarity(interaction_matrix)

    # Predict scores for unanswered questions
    predicted_scores = predict_scores(interaction_matrix, user_similarity)

    # Rank questions for each user based on predicted scores
    recommended_questions = rank_questions(predicted_scores, item_index)

    

    # Return ranked question recommendations
    return recommended_questions



@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    new_user_profile = data['new_user_profile']
    recommendations = get_recommendations(new_user_profile)
    return jsonify(recommendations)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=80)