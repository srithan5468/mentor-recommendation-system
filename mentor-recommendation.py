import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Generate Mock Data
def generate_mock_data():
    # Aspirant data
    subjects = ['Legal Reasoning', 'Logical Reasoning', 'English', 'Current Affairs', 'Quantitative Techniques']
    colleges = ['NLSIU Bangalore', 'NALSAR Hyderabad', 'NUJS Kolkata', 'NLU Delhi', 'NLIU Bhopal']
    prep_levels = ['Beginner', 'Intermediate', 'Advanced']
    learning_styles = ['Visual', 'Auditory', 'Reading/Writing', 'Kinesthetic']
    
    # Generate 50 aspirants
    np.random.seed(42)
    n_aspirants = 50
    
    aspirant_data = {
        'aspirant_id': [f'A{i:03d}' for i in range(1, n_aspirants+1)],
        'preferred_subject1': np.random.choice(subjects, n_aspirants),
        'preferred_subject2': np.random.choice(subjects, n_aspirants),
        'target_college1': np.random.choice(colleges, n_aspirants),
        'target_college2': np.random.choice(colleges, n_aspirants),
        'preparation_level': np.random.choice(prep_levels, n_aspirants),
        'learning_style': np.random.choice(learning_styles, n_aspirants),
        'hours_per_week': np.random.randint(10, 50, n_aspirants)
    }
    
    aspirants_df = pd.DataFrame(aspirant_data)
    
    # Mentor data
    n_mentors = 20
    mentor_data = {
        'mentor_id': [f'M{i:03d}' for i in range(1, n_mentors+1)],
        'name': [f'Mentor {i}' for i in range(1, n_mentors+1)],
        'strong_subject1': np.random.choice(subjects, n_mentors),
        'strong_subject2': np.random.choice(subjects, n_mentors),
        'alma_mater': np.random.choice(colleges, n_mentors),
        'teaching_style': np.random.choice(learning_styles, n_mentors),
        'years_experience': np.random.randint(1, 5, n_mentors),
        'clat_rank': np.random.randint(1, 100, n_mentors)
    }
    
    mentors_df = pd.DataFrame(mentor_data)
    
    return aspirants_df, mentors_df

# 2. Feature Engineering
def feature_engineering(aspirants_df, mentors_df):
    # Categorical features to encode
    aspirant_cat_features = ['preferred_subject1', 'preferred_subject2', 'target_college1', 
                             'target_college2', 'preparation_level', 'learning_style']
    mentor_cat_features = ['strong_subject1', 'strong_subject2', 'alma_mater', 'teaching_style']
    
    # Initialize encoders
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
    # Encode aspirant features
    aspirant_encoded = encoder.fit_transform(aspirants_df[aspirant_cat_features])
    aspirant_encoded_df = pd.DataFrame(
        aspirant_encoded, 
        columns=encoder.get_feature_names_out(aspirant_cat_features),
        index=aspirants_df.index
    )
    
    # Add numerical features
    aspirant_encoded_df['hours_per_week'] = MinMaxScaler().fit_transform(
        aspirants_df[['hours_per_week']])
    
    # Encode mentor features
    mentor_encoded = encoder.transform(mentors_df[mentor_cat_features])
    mentor_encoded_df = pd.DataFrame(
        mentor_encoded,
        columns=encoder.get_feature_names_out(mentor_cat_features),
        index=mentors_df.index
    )
    
    # Add numerical features
    scaler = MinMaxScaler()
    mentor_encoded_df['years_experience'] = scaler.fit_transform(
        mentors_df[['years_experience']])
    
    # Invert clat_rank (lower is better)
    mentor_encoded_df['clat_rank'] = scaler.fit_transform(
        -mentors_df[['clat_rank']].values.reshape(-1, 1))
    
    return aspirant_encoded_df, mentor_encoded_df, encoder

# 3. Create Recommendation System
def recommend_mentors(aspirant_id, aspirants_df, mentors_df, aspirant_features, mentor_features, top_n=3):
    # Find the aspirant
    aspirant_idx = aspirants_df.index[aspirants_df['aspirant_id'] == aspirant_id].tolist()[0]
    
    # Create feature vectors
    aspirant_vector = aspirant_features.iloc[aspirant_idx].values.reshape(1, -1)
    
    # Calculate similarity
    similarity_scores = cosine_similarity(aspirant_vector, mentor_features)
    
    # Get top N mentors
    top_indices = similarity_scores[0].argsort()[::-1][:top_n]
    
    # Map back to mentor IDs
    recommendations = mentors_df.iloc[top_indices]
    
    return recommendations

# 4. Incorporate Feedback System
class FeedbackSystem:
    def __init__(self, aspirants_df, mentors_df):
        self.feedback_data = pd.DataFrame(columns=['aspirant_id', 'mentor_id', 'rating', 'timestamp'])
        self.aspirants_df = aspirants_df
        self.mentors_df = mentors_df
        
    def add_feedback(self, aspirant_id, mentor_id, rating):
        """Add a feedback rating (1-5) from an aspirant for a mentor"""
        new_feedback = pd.DataFrame({
            'aspirant_id': [aspirant_id],
            'mentor_id': [mentor_id],
            'rating': [rating],
            'timestamp': [pd.Timestamp.now()]
        })
        self.feedback_data = pd.concat([self.feedback_data, new_feedback], ignore_index=True)
        
    def get_mentor_average_rating(self, mentor_id):
        """Get the average rating for a mentor"""
        mentor_feedback = self.feedback_data[self.feedback_data['mentor_id'] == mentor_id]
        if len(mentor_feedback) == 0:
            return None
        return mentor_feedback['rating'].mean()
    
    def get_mentors_by_rating(self, min_rating=4.0):
        """Get all mentors with average rating above threshold"""
        mentor_ratings = self.feedback_data.groupby('mentor_id')['rating'].mean().reset_index()
        return mentor_ratings[mentor_ratings['rating'] >= min_rating]
    
    def adjust_recommendations(self, recommendations, min_rating=3.5):
        """Adjust recommendations based on feedback ratings"""
        rated_mentors = []
        for _, mentor in recommendations.iterrows():
            mentor_id = mentor['mentor_id']
            rating = self.get_mentor_average_rating(mentor_id)
            if rating is not None and rating >= min_rating:
                rated_mentors.append((mentor_id, rating))
                
        # If we have rated mentors, prioritize them
        if rated_mentors:
            rated_mentors.sort(key=lambda x: x[1], reverse=True)
            top_mentor_ids = [mentor_id for mentor_id, _ in rated_mentors[:len(recommendations)]]
            return self.mentors_df[self.mentors_df['mentor_id'].isin(top_mentor_ids)]
        
        # Otherwise return original recommendations
        return recommendations
    
    def visualize_feedback_trends(self):
        """Visualize mentor ratings over time"""
        if len(self.feedback_data) < 5:
            print("Not enough feedback data for visualization")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Average rating trend over time
        self.feedback_data['date'] = self.feedback_data['timestamp'].dt.date
        avg_ratings = self.feedback_data.groupby('date')['rating'].mean()
        
        plt.subplot(1, 2, 1)
        avg_ratings.plot()
        plt.title('Average Mentor Ratings Over Time')
        plt.xlabel('Date')
        plt.ylabel('Average Rating')
        
        # Top mentors by rating
        plt.subplot(1, 2, 2)
        mentor_avg = self.feedback_data.groupby('mentor_id')['rating'].mean().sort_values(ascending=False).head(5)
        mentor_avg.plot(kind='bar')
        plt.title('Top 5 Mentors by Average Rating')
        plt.xlabel('Mentor ID')
        plt.ylabel('Average Rating')
        plt.tight_layout()
        
        plt.savefig('mentor_feedback_analysis.png')
        plt.close()

# 5. Main Execution Function
def main():
    # Generate data
    print("Generating mock data...")
    aspirants_df, mentors_df = generate_mock_data()
    
    # Process features
    print("Processing features...")
    aspirant_features, mentor_features, _ = feature_engineering(aspirants_df, mentors_df)
    
    # Initialize feedback system
    feedback_system = FeedbackSystem(aspirants_df, mentors_df)
    
    # Demo: Recommend mentors for a specific aspirant
    aspirant_id = 'A001'
    print(f"\nRecommending mentors for Aspirant {aspirant_id}:")
    print(f"Aspirant profile: {aspirants_df[aspirants_df['aspirant_id'] == aspirant_id].iloc[0].to_dict()}")
    
    recommendations = recommend_mentors(
        aspirant_id, aspirants_df, mentors_df, aspirant_features, mentor_features)
    
    print("\nTop 3 recommended mentors:")
    for i, (_, mentor) in enumerate(recommendations.iterrows(), 1):
        print(f"{i}. {mentor['name']} (ID: {mentor['mentor_id']})")
        print(f"   Strengths: {mentor['strong_subject1']}, {mentor['strong_subject2']}")
        print(f"   Teaching style: {mentor['teaching_style']}")
        print(f"   CLAT rank: {mentor['clat_rank']}")
        print()
    
    # Demo: Add some feedback data
    print("Simulating feedback from aspirants...")
    feedback_system.add_feedback('A001', 'M005', 5)
    feedback_system.add_feedback('A002', 'M005', 4)
    feedback_system.add_feedback('A003', 'M005', 5)
    feedback_system.add_feedback('A001', 'M010', 3)
    feedback_system.add_feedback('A002', 'M010', 2)
    feedback_system.add_feedback('A003', 'M015', 4)
    
    # Show adjusted recommendations
    print("\nAdjusted recommendations based on feedback:")
    adjusted_recommendations = feedback_system.adjust_recommendations(recommendations)
    for i, (_, mentor) in enumerate(adjusted_recommendations.iterrows(), 1):
        rating = feedback_system.get_mentor_average_rating(mentor['mentor_id'])
        rating_info = f"Average Rating: {rating:.1f}" if rating else "No ratings yet"
        
        print(f"{i}. {mentor['name']} (ID: {mentor['mentor_id']}) - {rating_info}")
        print(f"   Strengths: {mentor['strong_subject1']}, {mentor['strong_subject2']}")
        print(f"   Teaching style: {mentor['teaching_style']}")
        print()
    
    # Generate visualization if there's enough data
    feedback_system.visualize_feedback_trends()
    print("\nAnalysis complete! Check 'mentor_feedback_analysis.png' for visualizations.")

if __name__ == "__main__":
    main()