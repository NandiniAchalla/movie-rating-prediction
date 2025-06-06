import pandas as pd

def export_movies_data():
    # Read the movies.csv file
    print("Reading movies data...")
    movies_df = pd.read_csv("movies.csv")
    
    # Select relevant columns
    movies_data = movies_df[['id', 'title', 'genres']]
    
    # Export to a new CSV file
    output_file = "exported_movies.csv"
    movies_data.to_csv(output_file, index=False)
    print(f"\nMovies data exported to {output_file}")
    
    # Print sample of exported data
    print("\nSample of exported movies:")
    print(movies_data.head(10))

if __name__ == "__main__":
    export_movies_data() 