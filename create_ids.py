import pandas as pd
import random
import os

print("Checking files in current directory:")
print(os.listdir('.'))

try:
    print("\nReading movies.csv...")
    movies_df = pd.read_csv('movies.csv')
    print("Successfully read movies.csv")
    print("Columns in movies.csv:", movies_df.columns.tolist())
    
    # Generate new movie ID (using the highest existing ID + 1)
    new_movie_id = movies_df['id'].max() + 1
    print(f"\nGenerated new movie ID: {new_movie_id}")
    
    # Create a new movie entry with only the required columns
    new_movie = {
        'id': new_movie_id,
        'original_title': 'Fida',
        'title': 'Fida',
        'genres': 'Drama|Romance',
        'original_language': 'en',
        'overview': 'A romantic drama movie',
        'popularity': 100.0,
        'release_date': '2024-01-01',
        'vote_average': 0.0,
        'vote_count': 0
    }
    
    # Add any missing columns from the original dataframe
    for col in movies_df.columns:
        if col not in new_movie:
            new_movie[col] = None
    
    print("\nNew movie data:")
    print(new_movie)
    
    try:
        print("\nReading ratings.csv...")
        ratings_df = pd.read_csv('ratings.csv')
        print("Successfully read ratings.csv")
        print("Columns in ratings.csv:", ratings_df.columns.tolist())
        
        # Generate new user ID (using the highest existing user ID + 1)
        new_user_id = ratings_df['userId'].max() + 1
        print(f"\nGenerated new user ID: {new_user_id}")
        
        # Create a new user rating
        new_rating = {
            'userId': new_user_id,
            'movieId': new_movie_id,
            'rating': 4.5,
            'timestamp': int(pd.Timestamp.now().timestamp())
        }
        
        print("\nNew rating data:")
        print(new_rating)
        
        # Save the new data
        print("\nSaving new entries...")
        
        # Add new movie to movies.csv
        movies_df = pd.concat([movies_df, pd.DataFrame([new_movie])], ignore_index=True)
        movies_df.to_csv('movies.csv', index=False)
        print("Saved new movie to movies.csv")
        
        # Add new rating to ratings.csv
        ratings_df = pd.concat([ratings_df, pd.DataFrame([new_rating])], ignore_index=True)
        ratings_df.to_csv('ratings.csv', index=False)
        print("Saved new rating to ratings.csv")
        
        print("\nSuccessfully saved all new entries!")
        
    except Exception as e:
        print(f"Error with ratings.csv: {str(e)}")
        print("Only saving the new movie entry...")
        movies_df = pd.concat([movies_df, pd.DataFrame([new_movie])], ignore_index=True)
        movies_df.to_csv('movies.csv', index=False)
        print("Saved new movie to movies.csv")
        
except Exception as e:
    print(f"Error: {str(e)}")
    print("\nPlease check that movies.csv exists and has the correct format.") 