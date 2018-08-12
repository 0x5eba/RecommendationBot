import pandas as pd
import numpy as np
from ast import literal_eval
import csv
from movies_cinema_imdb import geprintt_global_md, get_global_links, get_global_indices_map

path = '../the-movies-dataset/'

def get_row_title(title):
    md = get_global_md()
    return md.loc[md['title'] == title]

def get_titles():
    md = get_global_md()
    return [str(t) for t in md['title']]

def get_most_poular():
    md = get_global_md()

    vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.95)

    qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres', 'poster_path']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')

    def weighted_rating(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)

    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)

    movie = []
    for i in qualified.head(7).values:
        movie.append([str(i[0]), "https://image.tmdb.org/t/p/original/" + str(i[-2])])
    return movie

def user_gender(userId):
    users = pd.read_csv(path + 'pop_new_users.csv')
    usernames = list(map(str, users['userId'].values))
    exist = str(userId) in usernames
    if exist:
        gender = [str(i) for i in users['gender'][users['userId'] == userId]][0]
        return gender
    return None

def add_user(userId, gender, age):
    print("new user " + str(userId))
    with open(path + 'pop_new_users.csv', 'a') as csvfile:
        fieldnames = ['userId','gender', 'age']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'userId':userId, 'gender':gender, 'age':age})


def add_rating(userId, movie_title, rating):
    md = get_global_md()
    indices_map = get_global_indices_map()

    print(str(userId) + " " + str(movie_title) + " " + str(rating))

    # links = get_global_links()
    # id_map = pd.read_csv(path_dest + 'links.csv')[['movieId', 'tmdbId']]
    # links = links[links['tmdbId'].notnull()]['tmdbId'].astype('int')

    # smd = md[md['id'].isin(links)]
    # indices = pd.Series(smd.index, index=smd['title'])

    # def convert_int(x):
    #     try:
    #         return int(x)
    #     except:
    #         return np.nan

    # id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
    # id_map.columns = ['movieId', 'id']
    # id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
    # indices_map = id_map.set_index('id')

    with open(path + 'pop_new_ratings.csv', 'a') as csvfile:
        fieldnames = ['useless', 'userId','movieId', 'rating']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        tmdbId = md.loc[md['title'] == movie_title]['id']
        tmdbId = tmdbId.values[0]
        movieId = indices_map['movieId'][tmdbId]

        writer.writerow({'useless':0, 'userId':userId, 'movieId':movieId, 'rating':rating})