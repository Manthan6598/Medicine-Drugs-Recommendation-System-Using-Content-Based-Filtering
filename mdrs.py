#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


# In[3]:


df_train = pd.read_csv('datasets/drugsComTrain_raw.tsv', sep='\t',usecols=['drugName','condition','review','rating'])
df_test = pd.read_csv('datasets/drugsComTest_raw.tsv', sep='\t',usecols=['drugName','condition','review','rating'])


# In[4]:


df = pd.concat([df_train, df_test])


# In[5]:


df.shape


# In[6]:


df.isnull().sum()*100/df.shape[0]


# In[7]:


df = df.dropna(subset=['review', 'drugName', 'condition','rating'])


# In[8]:


df.isnull().sum()


# In[9]:


df['review'] = df['review'].str.lower().str.replace('[^a-zA-Z0-9]', ' ', regex=True)


# In[10]:


df.head()


# In[11]:


vectorizer = TfidfVectorizer(stop_words='english',max_features=1000)
tfidf = vectorizer.fit_transform(df['review'])


# In[12]:


batch_size = 1000
n_batches = tfidf.shape[0] // batch_size + 1

# Compute cosine similarity on batches of reviews
cos_sim = []
for i in range(n_batches):
    start_idx = i * batch_size
    end_idx = min((i+1) * batch_size, tfidf.shape[0])
    batch_tfidf = tfidf[start_idx:end_idx]
    batch_cos_sim = cosine_similarity(batch_tfidf)
    cos_sim.append(batch_cos_sim)
cos_sim = np.array(cos_sim, dtype=object)


# In[13]:


def recommend_drugs(condition, top_n=10):
    # Filter the train data for the given condition
    condition_df = df[df['condition'] == condition]
    if len(condition_df) == 0:
        print(f"No reviews found for condition '{condition}'")
        return
    
    # Compute the average rating for each drug in the train set
    drug_ratings = condition_df.groupby('drugName')['rating'].mean()
    
    # Compute the average cosine similarity for each drug in the train set
    drug_cos_sim = []
    for drug in condition_df['drugName'].unique():
        drug_reviews = condition_df[condition_df['drugName'] == drug]['review']
        drug_tfidf = vectorizer.transform(drug_reviews)
        drug_cos_sim.append(cosine_similarity(tfidf, drug_tfidf).mean())
    drug_cos_sim = pd.Series(drug_cos_sim, index=condition_df['drugName'].unique())
    
    # Compute a combined score for each drug based on rating and similarity in the train set
    drug_scores = drug_ratings * (1 + drug_cos_sim)
    drug_scores = drug_scores.sort_values(ascending=False)
    top_drugs_train = [{'drug': drug, 'score': score, 'rating': drug_ratings[drug]} for drug, score in drug_scores.head(top_n).items()]
    return top_drugs_train


# In[14]:


df = df[~df['condition'].str.contains('</span>')]


# In[15]:


conditions = sorted(df['condition'].unique().tolist())


# In[16]:


type(conditions)


# In[18]:


import pickle
with open('conditions.pkl', 'wb') as f:
    pickle.dump(conditions, f)

