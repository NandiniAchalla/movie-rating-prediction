from flask import Flask, render_template, request, jsonify
from pyspark_ml_project import load_and_preprocess_data, create_features
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Load and preprocess data
print("Loading and preprocessing data...")
movie_ratings = load_and_preprocess_data()
enhanced_data = create_features(movie_ratings)

# Initialize user features dictionary and models
user_features = {}
reg_model = None
clf_model = None
scaler = None
feature_cols = ['num_ratings', 'rating_stddev', 'user_num_ratings', 'user_avg_rating', 'num_genres']

def train_models():
    global reg_model, clf_model, scaler
    
    print("Training models...")
    # Prepare feature matrix
    X = enhanced_data[feature_cols].fillna(0)
    y_reg = enhanced_data['rating'].fillna(enhanced_data['rating'].mean())
    y_clf = (y_reg >= 4).astype(int)
    
    # Split data
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train regression model
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train_scaled, y_reg_train)
    
    # Train classification model
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train_scaled, y_clf_train)
    
    # Calculate metrics
    reg_pred = reg_model.predict(X_test_scaled)
    clf_pred = clf_model.predict(X_test_scaled)
    clf_prob = clf_model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'regression': {
            'RMSE': np.sqrt(mean_squared_error(y_reg_test, reg_pred)),
            'MAE': mean_absolute_error(y_reg_test, reg_pred),
            'R2': r2_score(y_reg_test, reg_pred)
        },
        'classification': {
            'Accuracy': accuracy_score(y_clf_test, clf_pred),
            'AUC': roc_auc_score(y_clf_test, clf_prob),
            'F1': f1_score(y_clf_test, clf_pred)
        }
    }
    
    return metrics

# Train models on startup
print("Initial model training...")
model_metrics = train_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Try to get data from both JSON and form data
        if request.is_json:
            data = request.json
        else:
            data = request.form
            
        movie_name = data.get('movieName', '')
        username = data.get('username', '')
        
        if not movie_name or not username:
            return jsonify({
                'error': 'Please provide both movie name and username.'
            })
        
        # Find movie by name (case-insensitive and partial match)
        movie_mask = enhanced_data['title'].str.lower().str.contains(movie_name.lower(), na=False)
        movie_match = enhanced_data[movie_mask]
        
        if movie_match.empty:
            return jsonify({
                'error': f'Movie "{movie_name}" not found. Please check the spelling and try again.'
            })
        
        # Get movie features
        movie_features = movie_match.iloc[0]
        
        # Create or get user features
        if username not in user_features:
            user_features[username] = {
                'user_num_ratings': 0,
                'user_avg_rating': 3.5
            }
        user_feature = user_features[username]
        
        # Prepare features
        X = pd.DataFrame([{
            'num_ratings': float(movie_features.get('num_ratings', 0)),
            'rating_stddev': float(movie_features.get('rating_stddev', 0)),
            'user_num_ratings': float(user_feature['user_num_ratings']),
            'user_avg_rating': float(user_feature['user_avg_rating']),
            'num_genres': float(movie_features.get('num_genres', 1))
        }])
        
        # Scale features
        X_scaled = scaler.transform(X.fillna(0))
        
        # Make predictions
        rating_pred = reg_model.predict(X_scaled)[0]
        high_rating_prob = clf_model.predict_proba(X_scaled)[0][1]
        
        return jsonify({
            'movie_title': movie_features['title'],
            'predicted_rating': float(rating_pred),
            'high_rating_probability': float(high_rating_prob)
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your request. Please try again.'
        })

@app.route('/visualizations')
def visualizations():
    create_visualizations()
    return render_template('visualizations.html')

@app.route('/metrics')
def metrics():
    return render_template('metrics.html',
                         regression_metrics=model_metrics['regression'],
                         classification_metrics=model_metrics['classification'])

def create_visualizations():
    try:
        # Rating distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(movie_ratings['rating'].dropna(), bins=10)
        plt.title('Rating Distribution')
        plt.savefig('static/rating_distribution.png')
        plt.close()
        
        # Genre distribution
        plt.figure(figsize=(10, 6))
        genre_counts = movie_ratings['genres'].str.split('|').explode().value_counts()
        sns.barplot(x=genre_counts.values[:10], y=genre_counts.index[:10])
        plt.title('Top 10 Genres Distribution')
        plt.tight_layout()
        plt.savefig('static/genre_distribution.png')
        plt.close()
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True) 