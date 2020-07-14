#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


credits = pd.read_csv("tmdb_5000_credits.csv")


# In[3]:


movies = pd.read_csv("tmdb_5000_movies.csv")


# In[4]:


credits.head()


# In[5]:


movies.head()


# In[6]:


print("credits" , credits.shape)
print("movies", movies.shape)


# In[7]:


credits_colun_renamed = credits.rename(index=str, columns={"movie_id" = "id"})


# In[8]:


credits_colun_renamed = credits.rename(index=str, columns={"movie_id" : "id"})


# In[9]:


movies_merge = movies.merge(credits_colun_renamed, on ='id')
movies_merge.head()


# In[10]:


movies_cleaned_df = movies_merge.drop(columns = ['homepage', 'title_x', 'title_y', 'status', 'production_countries'])
movies_cleaned_df.head()


# In[11]:


movies_cleaned_df.info()


# In[13]:


movies_cleaned_df.head(1)['overview']


# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[19]:


tfv = TfidfVectorizer(min_df = 3, max_features = None,
                     strip_accents= 'unicode', analyzer = 'word',token_pattern = r'\w{1,}',
                     ngram_range = (1 , 3),
                     stop_words = 'english')
# Filling NANs with Empty String
movies_cleaned_df['overview'] = movies_cleaned_df['overview'].fillna('')


# In[20]:


# filling the TF-IDF on the overview text
tfv_matrix = tfv.fit_transform(movies_cleaned_df['overview'])


# In[21]:


tfv_matrix


# In[22]:


tfv_matrix.shape


# In[25]:


from sklearn.metrics.pairwise import sigmoid_kernel

# compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)


# In[26]:


sig[0]


# In[28]:


# Reverse mapping of indices and movi titles
indices = pd.Series(movies_cleaned_df.index, index=movies_cleaned_df['original_title']).drop_duplicates()


# In[29]:


indices.head()


# In[30]:


indices['Newlyweds']


# In[31]:


sig[4799]


# In[32]:


list(enumerate(sig[indices['Newlyweds']]))


# In[34]:


sorted(list(enumerate(sig[indices['Newlyweds']])), key = lambda x: x[1], reverse=True)


# In[41]:


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
    


# In[42]:


give_rec('Mr. 3000')


# In[ ]:




