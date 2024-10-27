import streamlit as st
import pandas as pd
from joblib import load
from surprise import Dataset, Reader
from surprise import SVD

# Load the saved model
model = load("book_recommender_model.joblib")

# Load the datasets
books = pd.read_csv("arashnic/book-recommendation-dataset/versions/3/Books.csv", low_memory=False)
ratings = pd.read_csv("arashnic/book-recommendation-dataset/versions/3/Ratings.csv")
users = pd.read_csv("arashnic/book-recommendation-dataset/versions/3/Users.csv")

# Merge datasets for filtering options
data = pd.merge(ratings, users, on="User-ID").merge(books, on="ISBN")


# Streamlit user interface
st.title("Book Recommendation System")
st.sidebar.header("Filter Options")

# Filter by Age

age_filter = st.sidebar.slider(
    "Age", 
    min_value=int(users['Age'].min()), 
    max_value=80,  # Set maximum age to 80
    value=int(users['Age'].mean())
)


# Filter by Location
location_filter = st.sidebar.text_input("Location")

# Filter by Rating
min_rating = st.sidebar.slider("Minimum Rating", 1, 10, 5)

# Apply filters to dataset
filtered_data = data[(data['Age'] >= age_filter) & (data['Location'].str.contains(location_filter, case=False)) & (data['Book-Rating'] >= min_rating)]

st.write("### Filtered Books")
st.write(filtered_data[['Book-Title', 'Book-Author', 'Publisher', 'Year-Of-Publication']])

# Recommendations based on user filter
user_id = st.text_input("Enter User ID for Personalized Recommendations")
if st.button("Get Recommendations"):
    try:
        # Getting books not yet rated by the user for recommendation
        rated_books = set(filtered_data[filtered_data['User-ID'] == int(user_id)]['Book-Title'])
        all_books = set(data['Book-Title'])
        books_to_predict = list(all_books - rated_books)

        recommendations = []
        for book in books_to_predict:
            pred = model.predict(int(user_id), book)
            recommendations.append((book, pred.est))

        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]
        recommended_books = [rec[0] for rec in recommendations]
        st.write("### Recommended Books")
        st.write(recommended_books)
    except ValueError:
        st.write("User ID not found in dataset")
