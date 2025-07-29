"""
Movie Recommendation Engine using TensorFlow and MovieLens dataset
"""

import os
import zipfile
import requests
import pandas as pd
import numpy as np

def download_and_extract_movielens():
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    dataset_path = "ml-latest-small"
    if not os.path.exists(dataset_path):
        print("Downloading MovieLens dataset...")
        r = requests.get(url)
        with open("ml-latest-small.zip", "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile("ml-latest-small.zip", "r") as zip_ref:
            zip_ref.extractall()
        print("Dataset downloaded and extracted.")
    else:
        print("Dataset already exists.")
    return dataset_path


def main():
    dataset_path = download_and_extract_movielens()
    ratings = pd.read_csv(f"{dataset_path}/ratings.csv")

    # Step 2: Preprocess data
    user_ids = ratings['userId'].unique().tolist()
    movie_ids = ratings['movieId'].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    ratings['user'] = ratings['userId'].map(user2user_encoded)
    ratings['movie'] = ratings['movieId'].map(movie2movie_encoded)
    num_users = len(user2user_encoded)
    num_movies = len(movie2movie_encoded)
    print(f"Number of users: {num_users}, Number of movies: {num_movies}")

    # Prepare training data
    x = ratings[['user', 'movie']].values
    y = ratings['rating'].values

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Step 3: Build the model
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    embedding_size = 100

    user_input = keras.Input(shape=(1,), name='user')
    movie_input = keras.Input(shape=(1,), name='movie')

    user_embedding = layers.Embedding(num_users, embedding_size, name='user_embedding')(user_input)
    movie_embedding = layers.Embedding(num_movies, embedding_size, name='movie_embedding')(movie_input)

    dot_product = layers.Dot(axes=2)([user_embedding, movie_embedding])
    dot_product = layers.Flatten()(dot_product)

    output = layers.Dense(1, activation='linear')(dot_product)

    # This is a simple neural network (matrix factorization with embeddings)
    model = keras.Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    # Step 4: Train the model
    print("Training the model...")
    model.fit([x_train[:, 0], x_train[:, 1]], y_train, batch_size=64, epochs=20, verbose=1, validation_split=0.1)

  
    test_loss, test_mae = model.evaluate([x_test[:, 0], x_test[:, 1]], y_test, verbose=1)
    print(f"Test MAE: {test_mae}")


    # Step 6: Make recommendations for a user
    user_id = 1  # Example user
    user_encoded = user2user_encoded[user_id]
    movies_watched = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    movies_not_watched = [m for m in movie_ids if m not in movies_watched]
    movies_not_watched_encoded = [movie2movie_encoded[m] for m in movies_not_watched]
    user_array = np.array([user_encoded for _ in movies_not_watched_encoded])
    movie_array = np.array(movies_not_watched_encoded)
    predictions = model.predict([user_array, movie_array])
    top_indices = predictions.flatten().argsort()[-5:][::-1]
    top_movie_ids = [movies_not_watched[i] for i in top_indices]
    movies = pd.read_csv(f"{dataset_path}/movies.csv")
    print("Top 5 movie recommendations for user 1:")
    print(movies[movies['movieId'].isin(top_movie_ids)][['title']])

    # Step 7: Evaluate recommendation quality (Hit Rate@5)
    print("\nEvaluating Top-5 Recommendation Hit Rate on test users...")
    test_users = np.unique(x_test[:, 0])
    max_test_users = 50  # Limit for speed
    hits = 0
    total = 0
    for idx, test_user in enumerate(test_users[:max_test_users]):
        # Get all test movies for this user
        test_indices = np.where(x_test[:, 0] == test_user)[0]
        if len(test_indices) == 0:
            continue
        # Pick one test movie as ground truth
        gt_movie = x_test[test_indices[0], 1]
        # Recommend top-5 movies for this user (excluding watched)
        user_id_inv = user_ids[int(test_user)]
        watched = ratings[ratings['userId'] == user_id_inv]['movieId'].tolist()
        not_watched = [m for m in movie_ids if m not in watched]
        not_watched_encoded = [movie2movie_encoded[m] for m in not_watched]
        user_array = np.array([test_user for _ in not_watched_encoded])
        movie_array = np.array(not_watched_encoded)
        preds = model.predict([user_array, movie_array], verbose=0)
        top5_idx = preds.flatten().argsort()[-5:][::-1]
        top5_movies = [not_watched_encoded[i] for i in top5_idx]
        if int(gt_movie) in top5_movies:
            hits += 1
        total += 1
        if (idx+1) % 10 == 0:
            print(f"Evaluated {idx+1} users...")
    hit_rate = hits / total if total > 0 else 0
    print(f"Hit Rate@5 (first {total} test users): {hit_rate:.3f} ({hits}/{total})")


if __name__ == "__main__":
    main()
