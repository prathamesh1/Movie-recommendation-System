# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:04:35 2020

@author: PrathameshSirsikar
"""
import pandas as pd
import numpy as np

credits = pd.read_csv("tmdb_5000_credits.csv")
movies = pd.read_csv("tmdb_5000_movies.csv")
credits.head()
movies.head()
print("credits" , credits.shape)
print("movies", movies.shape)

credits_colun_renamed = credits.rename(index=str, columns={"movie_id" : "id"})
movies_merge = movies.merge(credits_colun_renamed, on ='id')
movies_merge.head()

movies_cleaned_df = movies_merge.drop(columns = ['homepage', 'title_x', 'title_y', 'status', 'production_countries'])
movies_cleaned_df.head()

movies_cleaned_df.info()
movies_cleaned_df.head(1)['overview']


from sklearn.feature_extraction.text import TfidfVectorizer
tfv = TfidfVectorizer(min_df = 3, max_features = None,
                     strip_accents= 'unicode', analyzer = 'word',token_pattern = r'\w{1,}',
                     ngram_range = (1 , 3),
                     stop_words = 'english')
# Filling NANs with Empty String
movies_cleaned_df['overview'] = movies_cleaned_df['overview'].fillna('')

# filling the TF-IDF on the overview text
tfv_matrix = tfv.fit_transform(movies_cleaned_df['overview'])

tfv_matrix

from sklearn.metrics.pairwise import sigmoid_kernel

# compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

# Reverse mapping of indices and movi titles
indices = pd.Series(movies_cleaned_df.index, index=movies_cleaned_df['original_title']).drop_duplicates()
list(enumerate(sig[indices['Newlyweds']]))

sorted(list(enumerate(sig[indices['Newlyweds']])), key = lambda x: x[1], reverse=True)


def give_rec(title, sig=sig):
    # Get the index corrosponding to original_title
    idx = indices[title]
    
    # Get the pairwise similarity scores
    sig_scores = list(enumerate(sig[idx]))
    
    # Sort the movies
    sig_scores = sorted(sig_scores, key = lambda x:x[1], reverse=True)
    
    # Scores of the top 10 most similar movies
    sig_scores = sig_scores[1:11]
    
    # movie_indices
    movie_indices = [i[0] for i in sig_scores]
    
    #Top 10 most similar movies
    return movies_cleaned_df['original_title'].iloc[movie_indices]

give_rec('Mr. 3000')


#Saving Model to disk
import pickle
pickle.dump(give_rec, open('model.pkl','wb'))

model = pickle.load(open('model.pkl', 'rb'))

