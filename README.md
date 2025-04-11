# CLAT Mentor Recommendation System

## Project Overview
This project implements a machine learning-based recommendation system that matches CLAT (Common Law Admission Test) aspirants with suitable mentors based on their profiles, preferences, and learning styles. Created as part of the NLTI internship assignment.

## Features
- Generates and processes mock data for aspirants and mentors
- Uses feature engineering to prepare data for similarity matching
- Implements cosine similarity algorithm for mentor recommendations
- Includes a feedback system to improve recommendations over time
- Visualizes feedback trends and mentor performance

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Installation
1. Clone this repository:
git clone [https://github.com/your-username/mentor-recommendation-system.git](https://github.com/srithan5468/mentor-recommendation-system)
cd mentor-recommendation

2. Install required packages:
pip install -r requirements.txt

## Usage
Run the main script:
python mentor_recommendation.py

The script will:
1. Generate mock data for aspirants and mentors
2. Process and engineer features
3. Recommend mentors for a sample aspirant
4. Simulate feedback and show how recommendations adjust
5. Generate visualizations of feedback trends

## How It Works
1. **Data Generation**: Creates mock profiles for aspirants and mentors with relevant attributes
2. **Feature Engineering**: Transforms categorical data using one-hot encoding and normalizes numerical features
3. **Recommendation Algorithm**: Uses cosine similarity to find mentors most similar to an aspirant's profile
4. **Feedback System**: Collects ratings and adjusts recommendations based on performance
5. **Visualization**: Shows trends in mentor ratings over time

## Improvement Possibilities
- Add more granular subject preferences and specializations
- Implement weighted features for more precise matching
- Use collaborative filtering with aspirant clusters
- Expand feedback to include topic-specific ratings
- Incorporate time decay for feedback relevance

## Project Structure
- `mentor_recommendation.py`: Main script with all functionality
- `requirements.txt`: List of required Python packages
- `README.md`: Documentation and instructions

## Evaluation Criteria
- Clarity and thoughtfulness of approach
- Code quality and documentation
- Creativity in applying ML concepts
- Relevance to law mentorship and aspirant support
