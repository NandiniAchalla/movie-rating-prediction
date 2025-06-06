import pandas as pd
import numpy as np
import json
from pyspark_ml_project import load_and_preprocess_data, create_features, train_models

def add_new_rating(username, movie_name, rating):
    print(f"\nAdding new rating for user '{username}' for movie '{movie_name}'...")
    
    # Load existing data
    movie_ratings = load_and_preprocess_data()
    
    # Create new user ID (max existing ID + 1)
    new_user_id = movie_ratings['userId'].max() + 1
    
    # Try to find movie ID
    movie_match = movie_ratings[movie_ratings['title'].str.contains(movie_name, case=False, na=False)]
    
    if len(movie_match) == 0:
        print(f"Movie '{movie_name}' not found in database.")
        return None
    
    movie_id = movie_match.iloc[0]['movieId']
    movie_title = movie_match.iloc[0]['title']
    
    # Create new rating entry
    new_rating = pd.DataFrame({
        'userId': [new_user_id],
        'movieId': [movie_id],
        'rating': [float(rating)],
        'timestamp': [int(pd.Timestamp.now().timestamp())],
        'username': [username]
    })
    
    print(f"\nNew rating added successfully:")
    print(f"Username: {username} (User ID: {new_user_id})")
    print(f"Movie: {movie_title}")
    print(f"Rating: {rating}/5.0")
    
    # Update the data
    movie_ratings = pd.concat([movie_ratings, new_rating], ignore_index=True)
    
    return new_user_id, movie_title

def predict_multiple_ratings():
    # Load and process data
    print("Loading data...")
    movie_ratings = load_and_preprocess_data()
    enhanced_data = create_features(movie_ratings)
    
    # Train models
    print("Training model...")
    reg_model, scaler, metrics = train_models(enhanced_data)
    
    # Test combinations with actual ratings
    test_combinations = [
        (2, "The Dark Knight", 3.17),
        (4, "Jurassic Park", 3.31),
        (8, "Titanic", 2.81),
        (11, "Iron Man", 3.46),
        (12, "Star Wars", 4.09)
    ]
    
    print("\nMaking predictions for specific user-movie combinations:")
    print("-" * 50)
    
    predictions = []
    for user_id, movie_name, actual_rating in test_combinations:
        try:
            # Find movie ID
            movie_match = movie_ratings[movie_ratings['title'].str.contains(movie_name, case=False, na=False)]
            if len(movie_match) == 0:
                print(f"Movie '{movie_name}' not found in database.")
                continue
                
            movie_id = movie_match.iloc[0]['movieId']
            movie_title = movie_match.iloc[0]['title']
            
            # Get movie features
            movie_features = enhanced_data[
                (enhanced_data['movieId'] == movie_id)
            ].iloc[0]
            
            # Get user features
            user_features = enhanced_data[
                (enhanced_data['userId'] == user_id)
            ].iloc[0]
            
            # Prepare features for prediction
            features = {
                'num_ratings': [movie_features['num_ratings']],
                'rating_stddev': [movie_features['rating_stddev']],
                'user_num_ratings': [user_features['user_num_ratings']],
                'user_avg_rating': [user_features['user_avg_rating']],
                'num_genres': [movie_features['num_genres']]
            }
            
            # Convert features to JSON-serializable format
            features = {k: float(v[0]) for k, v in features.items()}
            
            # Scale features
            scaled_features = scaler.transform(pd.DataFrame([features]))
            
            # Make prediction
            prediction = float(reg_model.predict(scaled_features)[0])
            
            # Store prediction result
            result = {
                'user_id': user_id,
                'movie_title': movie_title,
                'predicted_rating': prediction,
                'actual_rating': actual_rating,
                'difference': abs(prediction - actual_rating)
            }
            predictions.append(result)
            
            # Print results
            print(f"User {user_id} - {movie_title}:")
            print(f"Predicted Rating: {prediction:.2f}/5.0")
            print(f"Actual Rating: {actual_rating}/5.0")
            print(f"Difference: {abs(prediction - actual_rating):.2f}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error predicting for User {user_id} - {movie_name}: {str(e)}")
            print("-" * 50)
    
    # Print model performance metrics
    metrics_json = {
        'rmse': float(metrics['rmse']),
        'mae': float(metrics['mae']),
        'r2': float(metrics['r2'])
    }
    
    print("\nModel Performance Metrics:")
    print(json.dumps(metrics_json, indent=2))
    
    return {'predictions': predictions, 'metrics': metrics_json}

def predict_single_rating(user_id, movie_name):
    # Load and process data
    print(f"Loading data for prediction of {movie_name}...")
    movie_ratings = load_and_preprocess_data()
    enhanced_data = create_features(movie_ratings)
    
    # Train models
    print("Training model...")
    reg_model, scaler, metrics = train_models(enhanced_data)
    
    try:
        # Get movie ID
        movie_id = movie_ratings[movie_ratings['title'] == movie_name]['movieId'].iloc[0]
        
        # Get features
        movie_features = enhanced_data[enhanced_data['movieId'] == movie_id].iloc[0]
        user_features = enhanced_data[enhanced_data['userId'] == user_id].iloc[0]
        
        # Create feature vector
        features = pd.DataFrame({
            'num_ratings': [movie_features['num_ratings']],
            'rating_stddev': [movie_features['rating_stddev']],
            'user_num_ratings': [user_features['user_num_ratings']],
            'user_avg_rating': [user_features['user_avg_rating']],
            'num_genres': [movie_features['num_genres']]
        })
        
        # Scale features and predict
        features_scaled = scaler.transform(features)
        predicted_rating = reg_model.predict(features_scaled)[0]
        
        # Get actual rating if it exists
        actual_rating = movie_ratings[
            (movie_ratings['userId'] == user_id) & 
            (movie_ratings['movieId'] == movie_id)
        ]['rating'].iloc[0] if len(movie_ratings[
            (movie_ratings['userId'] == user_id) & 
            (movie_ratings['movieId'] == movie_id)
        ]) > 0 else "Not rated"
        
        # Get movie genres
        movie_genres = movie_ratings[movie_ratings['movieId'] == movie_id]['genres'].iloc[0]
        
        print("\nPrediction Results:")
        print(f"User ID: {user_id}")
        print(f"Movie: {movie_name}")
        print(f"Genres: {movie_genres}")
        print(f"Predicted Rating: {predicted_rating:.2f}/5.0")
        print(f"Actual Rating: {actual_rating}")
        print("\nModel Performance Metrics:")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R2 Score: {metrics['r2']:.4f}")
        
    except (IndexError, KeyError) as e:
        print(f"Error: Could not find data for user {user_id} or movie '{movie_name}'")

def search_user_movie(user_id, movie_name):
    # Load and process data
    print(f"Searching for User {user_id} and movie '{movie_name}'...")
    movie_ratings = load_and_preprocess_data()
    enhanced_data = create_features(movie_ratings)
    
    try:
        # Get movie ID
        movie_id = movie_ratings[movie_ratings['title'] == movie_name]['movieId'].iloc[0]
        
        # Get features
        movie_features = enhanced_data[enhanced_data['movieId'] == movie_id].iloc[0]
        user_features = enhanced_data[enhanced_data['userId'] == user_id].iloc[0]
        
        # Create feature vector
        features = pd.DataFrame({
            'num_ratings': [movie_features['num_ratings']],
            'rating_stddev': [movie_features['rating_stddev']],
            'user_num_ratings': [user_features['user_num_ratings']],
            'user_avg_rating': [user_features['user_avg_rating']],
            'num_genres': [movie_features['num_genres']]
        })
        
        # Train model
        print("Training model...")
        reg_model, scaler, metrics = train_models(enhanced_data)
        
        # Scale features and predict
        features_scaled = scaler.transform(features)
        predicted_rating = reg_model.predict(features_scaled)[0]
        
        # Get actual rating if it exists
        actual_rating = movie_ratings[
            (movie_ratings['userId'] == user_id) & 
            (movie_ratings['movieId'] == movie_id)
        ]['rating'].iloc[0] if len(movie_ratings[
            (movie_ratings['userId'] == user_id) & 
            (movie_ratings['movieId'] == movie_id)
        ]) > 0 else "Not rated"
        
        # Get movie genres
        movie_genres = movie_ratings[movie_ratings['movieId'] == movie_id]['genres'].iloc[0]
        
        print("\nSearch Results:")
        print(f"User ID: {user_id}")
        print(f"Movie: {movie_name}")
        print(f"Genres: {movie_genres}")
        print(f"Predicted Rating: {predicted_rating:.2f}/5.0")
        print(f"Actual Rating: {actual_rating}")
        print("\nModel Performance Metrics:")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R2 Score: {metrics['r2']:.4f}")
        
    except (IndexError, KeyError) as e:
        print(f"Error: Could not find data for user {user_id} or movie '{movie_name}'")

if __name__ == "__main__":
    results = predict_multiple_ratings()
    # Output final results as JSON
    print("\nFinal Results:")
    print(json.dumps(results, indent=2)) 