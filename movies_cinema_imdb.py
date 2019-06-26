
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import csv
import tmdbsimple as tmdb
import time
import numpy as np
import datetime
import copy
from unidecode import unidecode
import calendar

from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate

# import sys
# from importlib import reload
# reload(sys)
# sys.setdefaultencoding('utf8')

tmdb.API_KEY = 'API_KEY'

path_src = '../the-movies-dataset/'
path_dest = '../cinema-dataset/'

final_list_recent_movies = []

def get_tmdb_info(title, original_year):
    search = tmdb.Search()
    response = search.movie(query=str(title))
    for res in search.results:
        year = res['release_date'].split('-')[0] 
        if str(year) == str(original_year):
            print(title)

            m = tmdb.Movies(res['id'])
            credits = m.credits()
            keywords = m.keywords()

            overview = ''.join([i if ord(i) < 128 else '~' for i in res['overview']])
            cast = ''.join([i if ord(i) < 128 else '~' for i in str(credits['cast'][:5])])
            crew = ''.join([i if ord(i) < 128 else '~' for i in str(credits['crew'][:5])])
            title = ''.join([i if ord(i) < 128 else '~' for i in res['title']])
            year = str(res['release_date']).split('-')[0]

            res = m.info()
            film = {'auseless':0, 'budget':res['budget'], 'genres':res['genres'], 'id': res['id'], 'imdb_id': res['imdb_id'],
                    'overview':overview, 'popularity':res['popularity'], 'poster_path':res['poster_path'], 'revenue':res['revenue'], 
                    'runtime':res['runtime'],'title':title, 'vote_average':res['vote_average'], 
                    'vote_count':res['vote_count'], 'year':year}

            links_csv = pd.read_csv(path_dest + 'links.csv')
            id_already_done = links_csv['tmdbId'].values
            for i in id_already_done:
                if int(str(res['id'])) == int(i):
                    print("already in links")
                    return title
            last_row_links = links_csv.tail(1)
            free_movieId = int(last_row_links['movieId'].values)+1
            
            
            with open(path_dest + 'metadata.csv', 'a') as csvfile:
                fieldnames = list(film.keys())
                fieldnames.sort()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(film)

            with open(path_dest + 'credits.csv', 'a') as csvfile:
                fieldnames = ['useless', 'cast','crew','id']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'useless':0, 'cast':cast, 'crew':crew, 'id':res['id']})
            
            with open(path_dest + 'keywords.csv', 'a') as csvfile:
                fieldnames = ['useless', 'id', 'keywords']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'useless':0, 'id':res['id'], 'keywords':keywords['keywords']})
            
            with open(path_dest + 'links.csv', 'a') as csvfile:
                fieldnames = ['useless', 'movieId', 'imdbId', 'tmdbId']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'useless':0, 'movieId':free_movieId, 'imdbId':str(res['imdb_id'])[2:], 'tmdbId':str(res['id'])+'.0'})
            
            print("done")
            time.sleep(1)
            return title

global_md, global_links, global_credits, global_keywords, global_cosine_sim, global_inverse_indices, global_indices_map, global_indices_map_for_tmdb, global_smd = None, None, None, None, None, None, None, None, None

def get_global_md():
    return global_md
def get_global_links():
    return global_links
def get_global_credits():
    return global_credits
def get_global_keywords():
    return global_keywords
def get_global_cosine_sim():
    return global_cosine_sim
def get_global_inverse_indices():
    return global_inverse_indices
def get_global_indices_map():
    return global_indices_map
def get_global_indices_map_for_tmdb():
    return global_indices_map_for_tmdb
def get_global_smd():
    return global_smd

def load_everything():
    md = pd.read_csv(path_dest + 'metadata.csv')
    links = pd.read_csv(path_dest + 'links.csv')
    credits = pd.read_csv(path_dest + 'credits.csv')
    keywords = pd.read_csv(path_dest + 'keywords.csv')
    del md['useless']
    del links['useless']
    del credits['useless']
    del keywords['useless']

    md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    md['popularity'] = md['popularity'].fillna('[]').apply(lambda x: [str(int(x))] if isinstance(x, float) or isinstance(x, int) else [])

    links = links[links['tmdbId'].notnull()]['tmdbId'].astype('int')
    md['id'] = md['id'].astype('int')
    smd = md[md['id'].isin(links)]

    keywords['id'] = keywords['id'].astype('int')
    credits['id'] = credits['id'].astype('int')
    md['id'] = md['id'].astype('int')

    md = md.merge(credits, on='id')
    md = md.merge(keywords, on='id')
    smd = md[md['id'].isin(links)]

    smd['cast'] = smd['cast'].apply(literal_eval)
    smd['crew'] = smd['crew'].apply(literal_eval)
    smd['keywords'] = smd['keywords'].apply(literal_eval)

    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    indices = pd.Series(smd.index, index=smd['title'])
    smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
    smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    smd['director'] = smd['crew'].apply(get_director)
    smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
    smd['director'] = smd['director'].apply(lambda x: [x,x,x])

    s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'keyword'
    s = s.value_counts()
    s = s[s > 1]
    stemmer = SnowballStemmer('english')
    stemmer.stem('dogs')

    def filter_keywords(x):
        words = []
        for i in x:
            if i in s:
                words.append(i)
        return words

    smd['keywords'] = smd['keywords'].apply(filter_keywords)
    smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

    smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres'] + smd['popularity'] # + smd['year'] 
    smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))

    count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    count_matrix = count.fit_transform(smd['soup'])

    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    smd = smd.reset_index()
    titles = smd['title']
    indices = pd.Series(smd.index, index=smd['title'])
    inverse_indices = pd.Series(smd['title'], index=smd.index)

    def convert_int(x):
        try:
            return int(x)
        except:
            return np.nan
    id_map = pd.read_csv(path_dest + 'links.csv')[['movieId', 'tmdbId']]
    id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
    id_map.columns = ['movieId', 'id']
    id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
    indices_map = id_map.set_index('id')
    indices_map_for_tmdb = id_map.set_index('movieId')

    global global_md, global_links, global_credits, global_keywords, global_cosine_sim, global_inverse_indices, global_indices_map, global_indices_map_for_tmdb, global_smd

    global_md = copy.deepcopy(md)
    links = pd.read_csv(path_dest + 'links.csv')
    global_links = copy.deepcopy(links)
    global_credits = copy.deepcopy(credits)
    global_keywords = copy.deepcopy(keywords)
    global_cosine_sim = copy.deepcopy(cosine_sim)
    global_inverse_indices = copy.deepcopy(inverse_indices)
    global_indices_map = copy.deepcopy(indices_map)
    global_indices_map_for_tmdb = copy.deepcopy(indices_map_for_tmdb)
    global_smd = copy.deepcopy(smd)


def get_recent_movies():
    while True:
        global final_list_recent_movies
        final_list_recent_movies = []
        # if False:
        # from shutil import copyfile
        # copyfile(path_src + 'pop_new_metadata.csv', path_dest + 'metadata.csv')
        # copyfile(path_src + 'pop_new_links.csv', path_dest + 'links.csv')
        # copyfile(path_src + 'credits.csv', path_dest + 'credits.csv')
        # copyfile(path_src + 'keywords.csv', path_dest + 'keywords.csv')

        now = datetime.datetime.now()
        year = now.year
        month = now.month
        if month - 2 < 1:
            month = 12 - (month - 2)
            year -= 1
        else:
            month -= 2
        data_finish = str(now.year)+"-"+str(now.month).zfill(2)+"-30" # 2017-01-01
        data_start = str(year)+"-"+str(month).zfill(2)+"-1"

        url = "https://www.imdb.com/search/title?release_date=" + str(data_start) + "," + str(data_finish) + "&languages=en&sort=num_votes,desc&page="

        for number_page in range(1, 4):
            print("PAGE: " + str(number_page))
            url_num = url + str(number_page)
            req = requests.get(url_num)
            data = req.text
            data = ''.join([i if ord(i) < 128 else '~' for i in data])
            soup = BeautifulSoup(data,"html.parser")
            for movie in soup.findAll('div', {'class':'lister-item mode-advanced'}):
                imdb_rate = float(movie.find('div', {'class':'inline-block ratings-imdb-rating'}).get('data-value'))
                metascore = movie.find('div', {'class':'inline-block ratings-metascore'})
                if not metascore:
                    continue
                metascore = int(str(metascore.text[:3]))
                if float(metascore/10) + imdb_rate < 12.0:
                    continue
                a = movie.find('div', {'class':"lister-item-content"}).find('h3',{'class':"lister-item-header"}).find('a')
                imdb_link = "https://www.imdb.com" + '/'.join(str(a.get('href')).split('/')[:-1])
                italian_title = a.text
                year = movie.find('div', {'class':"lister-item-content"}).find('h3',{'class':"lister-item-header"}).find('span', {'class':'lister-item-year text-muted unbold'}).text
                year = str(year)[1:5]

                req_info = requests.get(imdb_link + "/releaseinfo")
                data_info = req_info.text
                data_info = ''.join([i if ord(i) < 128 else '~' for i in data_info])
                soup_info = BeautifulSoup(data_info,"html.parser")
                names = soup_info.find('table', {'class':'subpage_data spEven2Col', 'id':'akas'})
                if not names:
                    continue
                original_name = str(italian_title)
                names = names.text.split('\n\n')
                name_found = False
                for n in names:
                    if len(n.split('\n')) != 2:
                        continue
                    state, name = n.split('\n')
                    if state == "UK" or state == "USA":
                        name_found = True
                        original_name = name
                        break
                if not name_found:
                    for n in names:
                        if len(n.split('\n')) != 2:
                            continue
                        state, name = n.split('\n')
                        if state == "(original title)":
                            original_name = name
                            break

                if '~' in str(original_name):
                    continue
                
                release_date_italy = soup_info.find('table', {'class':'subpage_data spFirst', 'id':'release_dates'})
                release_date_found = None
                for n in release_date_italy.text.split('\n\n'):
                    if len(n.split('\n')) != 2:
                        continue
                    state, release_date = n.split('\n')
                    if state == "Italy":
                        release_date_found = release_date
                        break

                available = True
                if release_date_found:
                    now = datetime.datetime.now()
                    release_date_found_days, release_date_found_month, release_date_found_year = release_date_found.split(' ')
                    if int(release_date_found_year) > int(now.year):
                        available = False
                    for month_idx in range(1, 13):
                        if str(calendar.month_name[month_idx]) == release_date_found_month:
                            if int(month_idx) > int(now.month):
                                available = False
                                break
                            if int(month_idx) == int(now.month) and int(release_date_found_days) > int(now.day):
                                available = False
                                break

                md = pd.read_csv(path_dest + 'metadata.csv')
                title_already_done = md['title'].values
                if str(original_name) in title_already_done and original_name != None:
                    if available:
                        final_list_recent_movies.append([str(original_name), ""])
                    else:
                        ry, rm, ra = release_date_found.split(' ')
                        final_list_recent_movies.append([str(original_name), ' '.join([ry, rm[:3], ra])])
                    print(original_name + "  already done")
                    continue

                title_found = get_tmdb_info(original_name, year)
                if title_found != None:
                    if available:
                        final_list_recent_movies.append([str(original_name), ""])
                    else:
                        ry, rm, ra = release_date_found.split(' ')
                        final_list_recent_movies.append([str(original_name), ' '.join([ry, rm[:3], ra])])

        load_everything()
        print("ready")
        time.sleep(int(np.random.randint(172800)+300))

    
from threading import Thread
Thread(target=get_recent_movies).start()

def get_all_cinema_movies():
    return final_list_recent_movies

def final_cinema_movies(userId):                                                 
    global final_list_recent_movies
    global global_cosine_sim, global_inverse_indices, global_indices_map, global_indices_map_for_tmdb, global_smd

    ratings = pd.read_csv(path_src + 'pop_new_ratings.csv')
    del ratings['useless']
    from hybrid import get_svd
    svd = get_svd()
    
    print("cinema for " + str(userId))

    def hybrid_recommandation(userId, title, svd):
        idx = 0
        for i, t in enumerate(global_inverse_indices.values):
            if t == title:
                idx = i
                break
        
        sim_scores = list(enumerate(global_cosine_sim[int(idx)]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:10]
        movie_indices = [i[0] for i in sim_scores]

        movies = global_smd.iloc[movie_indices][['title','id']]

        def pred(x):
            try:
                return svd.predict(userId, global_indices_map.loc[x]['movieId']).est
            except:
                return 0

        movies['recommanded'] = movies['id'].apply(pred)
        movies = movies.sort_values('recommanded', ascending=False)
        total_predict = 0
        for i in movies['recommanded'].values:
            total_predict += float(i)
        return total_predict

    dict_rank_movies = {}
    for m, available in final_list_recent_movies:
        total_predict = hybrid_recommandation(userId, m, svd)
        if available == "":
            dict_rank_movies[str(m)] = total_predict
        else:
            dict_rank_movies[str(m) + " [" + str(available) + "]"] = total_predict
    
    best_movie_sorted = sorted(dict_rank_movies.items(), key=lambda x: x[1], reverse=True)

    element_to_take = []
    count_not_exit_yet = 0
    count_exit_yet = 0
    for i, (title, predict) in enumerate(best_movie_sorted):
        if count_exit_yet >= 2:
            break
        if ("2018]" in str(title) or "2019]" in str(title)) and count_not_exit_yet < 3:
            count_not_exit_yet += 1
            element_to_take.append((title, predict))
        elif "2018]" not in str(title) and "2019]" not in str(title):
            count_exit_yet += 1
            element_to_take.append((title, predict))
    try:
        print(element_to_take)
    except:
        pass
    return element_to_take