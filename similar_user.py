
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation, cosine
from sklearn.metrics import pairwise_distances
import sys, os
from movies_cinema_imdb import get_global_links

path = '../the-movies-dataset/'

def get_similar_user(userId):
    ratings = pd.read_csv(path + 'pop_new_ratings.csv')
    links = get_global_links()
    users = list(pd.read_csv(path + 'pop_new_users.csv')['userId'].values)
    del ratings['useless']

    all_users = list(set(ratings['userId']))
    movies = list(set(links['movieId']))

    idx_movie = {}
    for i, movie in enumerate(movies):
        idx_movie[movie] = i

    rows = []
    c = 1
    user_idx = {}
    idx_user = {}
    for i, user in enumerate(all_users):
        if str(user).isdigit():
            continue
        user_idx[str(user)] = c
        idx_user[int(c)] = str(user)
        c += 1
        res = [0]*(len(movies))
        movies_user = [int(i) for i in ratings['movieId'][ratings['userId'] == str(user)]]
        rating_user = [int(i) for i in ratings['rating'][ratings['userId'] == str(user)]]
        for j,m in enumerate(movies_user):
            res[idx_movie[m]] = rating_user[j]
        rows.append(res)

    df = pd.DataFrame(rows, columns=movies)
    k=len(user_idx)-1
    metric='cosine' #'correlation' for Pearson correlation similaries, 'cosine'

    cosine_sim = 1-pairwise_distances(df, metric=metric)
    def findksimilarusers(user, ratings, metric=metric, k=k):
        user_id = user_idx[user]
        similarities=[]
        indices=[]
        model_knn = NearestNeighbors(metric=metric, algorithm='auto') 
        model_knn.fit(ratings)

        distances, indices = model_knn.kneighbors(ratings.iloc[user_id-1, :].values.reshape(1, -1), n_neighbors = k+1)
        similarities = 1-distances.flatten()

        print("Similarity for " + str(user))
        for i in range(0, len(indices.flatten())):
            if indices.flatten()[i]+1 == user_id:
                continue
            print('{0}: User {1}, with similarity of {2}'.format(i, idx_user[indices.flatten()[i]+1], similarities.flatten()[i]))

        for i in range(0, len(indices.flatten())):
            if indices.flatten()[i]+1 == user_id:
                continue
            similar_user = idx_user[indices.flatten()[i]+1]
            if similar_user not in users:
                continue
            return similar_user, str(int(float(str(similarities.flatten()[i])[:4])*100))
        return ""
            # print('{0}: User {1}, with similarity of {2}'.format(i, , similarities.flatten()[i]))
        # return similarities, indices

    return findksimilarusers(userId, df, metric=metric)