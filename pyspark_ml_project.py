import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import re

def load_and_preprocess_data():
    print("Loading data...")
    # Load data
    movies_df = pd.read_csv("movies.csv")
    ratings_df = pd.read_csv("ratings.csv")
    
    print("Processing movies data...")
    # Clean and process movies data
    movies_df['genres'] = movies_df['genres'].fillna('')
    movies_df['num_genres'] = movies_df['genres'].str.count('\|') + 1
    movies_df['num_genres'] = movies_df['num_genres'].fillna(0)
    
    print("Merging data...")
    # Merge datasets
    movie_ratings = pd.merge(ratings_df, movies_df[['id', 'title', 'genres', 'num_genres']], 
                           left_on='movieId', right_on='id', how='left')
    
    print("Data loaded successfully!")
    return movie_ratings

def create_features(movie_ratings):
    print("Creating features...")
    # Calculate movie statistics
    movie_stats = movie_ratings.groupby('movieId').agg({
        'rating': ['count', 'mean', 'std']
    }).reset_index()
    
    # Flatten column names
    movie_stats.columns = ['movieId', 'num_ratings', 'avg_rating', 'rating_stddev']
    
    # Fill NaN values in rating_stddev with 0
    movie_stats['rating_stddev'] = movie_stats['rating_stddev'].fillna(0)
    
    # Calculate user statistics
    user_stats = movie_ratings.groupby('userId').agg({
        'rating': ['count', 'mean']
    }).reset_index()
    
    # Flatten column names
    user_stats.columns = ['userId', 'user_num_ratings', 'user_avg_rating']
    
    # Join all statistics back to the original data
    enhanced_data = pd.merge(movie_ratings, movie_stats, on='movieId')
    enhanced_data = pd.merge(enhanced_data, user_stats, on='userId')
    
    print("Features created successfully!")
    return enhanced_data

def plot_distributions(data):
    print("Creating visualizations...")
    # Rating distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='rating', bins=10)
    plt.title('Rating Distribution')
    plt.savefig('static/rating_distribution.png')
    plt.close()
    
    # Genre distribution
    plt.figure(figsize=(12, 6))
    genre_counts = data['genres'].str.split('|').explode().value_counts()
    sns.barplot(x=genre_counts.values[:10], y=genre_counts.index[:10])
    plt.title('Top 10 Movie Genres')
    plt.tight_layout()
    plt.savefig('static/genre_distribution.png')
    plt.close()
    
    print("Visualizations created successfully!")

def classification_example():
    print("Running Classification Example...")
    
    # Load and preprocess data
    movie_ratings = load_and_preprocess_data()
    enhanced_data = create_features(movie_ratings)
    
    # Create binary target
    enhanced_data['high_rating'] = (enhanced_data['rating'] >= 4).astype(int)
    
    # Feature engineering
    feature_cols = ['num_ratings', 'avg_rating', 'rating_stddev', 
                   'user_num_ratings', 'user_avg_rating', 'num_genres']
    
    X = enhanced_data[feature_cols]
    y = enhanced_data['high_rating']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train multiple models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosted Trees": GradientBoostingClassifier()
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate
        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            "AUC": auc,
            "Accuracy": accuracy,
            "F1 Score": f1
        }
        
        print(f"{name} Results:")
        print(f"AUC: {auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    return results

def regression_example():
    print("\nRunning Regression Example...")
    
    # Load and preprocess data
    movie_ratings = load_and_preprocess_data()
    enhanced_data = create_features(movie_ratings)
    
    # Feature engineering
    feature_cols = ['num_ratings', 'rating_stddev', 
                   'user_num_ratings', 'user_avg_rating', 'num_genres']
    
    X = enhanced_data[feature_cols]
    y = enhanced_data['rating']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train multiple models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosted Trees": GradientBoostingRegressor()
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate using multiple metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }
        
        print(f"{name} Results:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")
    
    return results

def train_models(enhanced_data):
    print("Training models...")
    # Prepare features
    feature_cols = ['num_ratings', 'rating_stddev', 'user_num_ratings', 'user_avg_rating', 'num_genres']
    X = enhanced_data[feature_cols].fillna(0)
    y = enhanced_data['rating']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train regression model
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = reg_model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    print("Models trained successfully!")
    return reg_model, scaler, metrics

if __name__ == "__main__":
    # Load and process data
    movie_ratings = load_and_preprocess_data()
    enhanced_data = create_features(movie_ratings)
    
    # Create visualizations
    plot_distributions(enhanced_data)
    
    # Train models
    reg_model, scaler, metrics = train_models(enhanced_data)
    
    # Print metrics
    print("\nModel Performance Metrics:")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R2 Score: {metrics['r2']:.4f}")
    
    # Run classification and regression examples
    classification_results = classification_example()
    regression_results = regression_example() 