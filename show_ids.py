import pandas as pd
import os

print("Checking files in current directory:")
print(os.listdir('.'))

try:
    print("\nTrying to read movies.csv...")
    movies_df = pd.read_csv('movies.csv')
    print("Successfully loaded movies.csv")
    print("\nFirst few rows of movies data:")
    print(movies_df.head())
    
    print("\nColumns in movies.csv:")
    print(movies_df.columns.tolist())
    
    print("\nExample Movie IDs and Titles:")
    print(movies_df[['id', 'original_title']].head(10).to_string())
except Exception as e:
    print(f"Error reading movies.csv: {str(e)}")

try:
    print("\nTrying to read ratings.csv...")
    ratings_df = pd.read_csv('ratings.csv')
    print("Successfully loaded ratings.csv")
    print("\nFirst few rows of ratings data:")
    print(ratings_df.head())
    
    print("\nColumns in ratings.csv:")
    print(ratings_df.columns.tolist())
    
    print("\nExample User IDs and their ratings:")
    print(ratings_df[['userId', 'movieId', 'rating']].head(10).to_string())
except Exception as e:
    print(f"Error reading ratings.csv: {str(e)}") 