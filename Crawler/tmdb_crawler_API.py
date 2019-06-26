import tmdbsimple as tmdb
tmdb.API_KEY = 'API_KEY'

import pandas as pd
import numpy as np
import time, json


films = {}
    # try:
    #     movie = tmdb.Movies(i).info()
    # except:
    #     time.sleep(0.3)
    #     continue

search = tmdb.Search()
response = search.movie(query='Gli incredibili 2')
print(str(search.results).encode('utf-8'))
for data in search.results:
    data = ''.join([i if ord(i) < 128 else '~' for i in data['release_date']])
    print(data)

genres = [j['name'] for j in movie['genres']]
print(movie['release_date'])
year = str(movie['release_date']).split('-')[0]
films[movie['title']] ={'budget':movie['budget'], 'id': movie['id'], 'imdb_id': movie['imdb_id'], 'title':movie['title'], 'original_title':"",
                        'runtime':movie['runtime'], 'overview':movie['overview'], 'genres':genres, 'poster_path':movie['poster_path'],
                        'popularity':movie['popularity'], 'vote_average':movie['vote_average'], 
                        'vote_count':movie['vote_count'], 'year':year}

print(films[movie['title']])
exit(1)

if i % 1000 == 0:
    j = json.dumps(films)
    with open('/home/seba/DeepLearning/Recommandation-System/txt/tmdb_movie.txt', 'w') as f:
        f.write(j)

print(str(i) + " / " + m)
time.sleep(0.3)

j = json.dumps(films)
with open('/home/seba/DeepLearning/Recommandation-System/txt/tmdb_movie.txt', 'w') as f:
    f.write(j)

films = {}
with open('/home/seba/DeepLearning/Recommandation-System/txt/tmdb_movie.txt', 'r') as f:
    films = json.load(f)

tmdb_id, imdb_id, title, duration, genres, img, popularity, vote_average, vote_count, year_released =  [], [], [], [], [], [], [], [], [], []

for key, value in films.items():
    title.append(key)
    tmdb_id.append(value['tmdb_id'])
    imdb_id.append(value['imdb_id'])
    duration.append(value['duration'])
    genres.append(value['genres'])
    img.append(value['img'])
    popularity.append(value['popularity'])
    vote_average.append(value['vote_average'])
    vote_count.append(value['vote_count'])
    year_released.append(value['year_released'])
    

everything = [title, tmdb_id, imdb_id, duration, genres, img, popularity, vote_average, vote_count, year_released]

import csv
d = {'title':0, 'tmdb_id': 0, 'imdb_id': 0, 
    'duration':0, 'genres':0, 'img':0,
    'popularity':0, 'vote_average':0, 
    'vote_count':0, 'year_released':0}

with open('tmdb_csv.csv', 'w') as f:
    wr = csv.writer(f, dialect='excel')
    wr.writerow(d.keys())
    for i in range(len(everything[0])):
        tmp = []
        for j in range(len(everything)):
            tmp.append(str(everything[j][i]).encode('utf-8').lower())
        wr.writerow(tmp)

# f2 = open("altadefinizione01_csv.csv","w")
# with open('altadefinizione01.csv', 'r') as f:
#     for l in list(f):
#         f2.write(l.lower())

# search = tmdb.Search()
# response = search.movie(query='The Bourne')
# for s in search.results:
#      print(s['title'], s['id'], s['release_date'], s['popularity'])
