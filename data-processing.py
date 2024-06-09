import numpy as np
import pandas as pd
import pickle as pk
credits=pd.read_csv('tmdb_5000_credits.csv')
movies=pd.read_csv('tmdb_5000_movies.csv')
movies=movies.merge(credits,on='title')
movies=movies[['title','id','keywords','genres','overview','cast','crew']]
movies.isnull().sum()
movies.dropna(inplace=True)
movies.duplicated().sum()

import ast
def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l    
movies['genres']=movies['genres'].apply(convert)

movies['keywords']=movies['keywords'].apply(convert)

def convert2(obj):
    l=[]
    c=0
    for i in ast.literal_eval(obj):
        if c!=5:
            l.append(i['name'])
            c=c+1
        else:
            break    
    return l    
movies['cast']=movies['cast'].apply(convert2)

def finddirector(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l    
movies['crew']=movies['crew'].apply(finddirector)

movies['overview']=movies['overview'].apply(lambda x:x.split())
mov2=movies[['title','genres','cast','crew']]

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])

movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tag']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

dataframe=movies[['id','title','tag']]

df=dataframe.merge(mov2,on='title')
df['tag']=df['tag'].apply(lambda x:" ".join(x))

df['tag']=df['tag'].apply(lambda x:x.lower())

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000, stop_words='english')
vectors=cv.fit_transform(df['tag']).toarray()

import nltk
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
def stem(txt):
    y=[]
    for i in txt.split():
        y.append(ps.stem(i))
    return " ".join(y)

df['tag']=df['tag'].apply(stem)
stem('thriller')
from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)

def reccomend1(movie):
    movie = movie.lower()  # Convert input to lowercase
    index = df[df['title'].str.lower() == movie].index[0] 
    # index=df[df['title']==movie].index[0]
    d=similarity[index]
    l=sorted(list(enumerate(d)),reverse=True,key=lambda x:x[1])[0:6]
    for i in l:
        print(df.iloc[i[0]].title)

def reccomend2(movie):
    movie = movie.lower()  # Convert input to lowercase
    c=0
    for i in df['title']:
        if movie in df['title'].str.lower()[c]:
            movie=df['title'].str.lower()[c]
            break
        c+=1    
    index = df[df['title'].str.lower() == movie].index[0] 
    # index=df[df['title']==movie].index[0]
    d=similarity[index]
    l=sorted(list(enumerate(d)),reverse=True,key=lambda x:x[1])[0:6]
    for i in l:
        print(df.iloc[i[0]].title)
        
def reccomend3(movie):
    movie = movie.lower()  # Convert input to lowercase
    if movie in df['title'].str.lower().values:
        reccomend1(movie)
    else:
        reccomend2(movie)     


def recommend_combined(query):
    query = query.lower()  # Convert input to lowercase
    
    # Split the query into individual genres
    genres = query.split()
    
    # Initialize a DataFrame to store matching movies
    matched_movies = pd.DataFrame(columns=df.columns)
    
    # Iterate over each genre in the query
    for genre in genres:
        # Check if the query matches any genre exactly
        genre_exact_matches = df[df['genres'].apply(lambda x: genre in x)]
        
        # Check if the query matches partially with any genre
        genre_partial_matches = df[df['genres'].apply(lambda x: any(genre in g.lower() for g in x))]
        
        # Combine exact and partial matches
        genre_matches = pd.concat([genre_exact_matches, genre_partial_matches])
        
        # Add matches to the matched_movies DataFrame
        matched_movies = pd.concat([matched_movies, genre_matches])
    
    # Remove duplicates and sort by title
    matched_movies = matched_movies.drop_duplicates(subset='title')
    
    if not matched_movies.empty:
        print("Recommendations based on genres:", query)
        for title in matched_movies['title']:
            print(title)
    else:
        print("No matches found for genres:", query)


def recommend(query):
    query = query.lower()  # Convert input to lowercase
    
    # Check if the query matches any movie title exactly
    exact_matches = df[df['title'].str.lower() == query]
    
    # Check if the query matches partially with any movie title
    partial_matches = df[df['title'].str.lower().str.contains(query)]
    
    if not exact_matches.empty:
        print("Exact Movie Title Match:")
        reccomend3(query)
    elif not partial_matches.empty:
        print("Partial Movie Title Matches:")
        reccomend3(partial_matches.iloc[0]['title'])
    else:
        s = ['Western', 'Documentary', 'Thriller', 'Adventure', 'Animation', 'Romance', 'Action', 'Family', 'TV Movie', 'Fantasy', 'Comedy', 'History', 'Crime', 'War', 'Foreign', 'Horror', 'Drama', 'Mystery', 'Music']
        s = [genre.lower() for genre in s]
        x = query.split()
        all_present = all(item in s for item in x)
        if all_present:
            recommend_combined(query)
        else:         
            # Check if the query matches any actor name exactly
            actor_exact_matches = df[df['cast'].apply(lambda x: query in x)]
        
            # Check if the query matches partially with any actor name
            actor_partial_matches = df[df['cast'].apply(lambda x: any(query in actor.lower() for actor in x))]
        
            # Check if the query matches any genre exactly
            genre_exact_matches = df[df['genres'].apply(lambda x: query in x)]
        
            # Check if the query matches partially with any genre
            genre_partial_matches = df[df['genres'].apply(lambda x: any(query in genre.lower() for genre in x))]
            
            # Check if the query matches any actor name exactly
            crew_exact_matches = df[df['crew'].apply(lambda x: query in x)]
        
            # Check if the query matches partially with any actor name
            crew_partial_matches = df[df['crew'].apply(lambda x: any(query in crew.lower() for crew in x))]
            
            # Combine all matches
            all_matches = pd.concat([actor_exact_matches, actor_partial_matches, genre_exact_matches, genre_partial_matches, crew_partial_matches, crew_exact_matches])
        
            # Remove duplicates and sort by cosine similarity
            all_matches = all_matches.drop_duplicates(subset='title')
        
            if not all_matches.empty:
                print("Recommendations based on cast or genre:")
                for title in all_matches['title']:
                    print(title)
            else:
                print("No matches found.")

recommend('avatar')   # Recommend based on partial movie name

recommend('Orlando Bloom')  # Recommend based on actor name (exact match)

# recommend('swayz')  # Recommend based on partial actor name
# recommend('science fiction')  # Recommend based on genre name (exact match)
# recommend('action comedy')  # Recommend based on combination of genres
# recommend('nolan')  # Recommend based on combined genres



# import json

# with open('movies.json', 'w') as outfile:
#     json.dump(df.to_dict(), outfile)

# with open('similarity.json', 'w') as outfile:
#     json.dump(similarity.tolist(), outfile)


