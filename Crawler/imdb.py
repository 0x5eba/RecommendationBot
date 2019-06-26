
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import csv
import tmdbsimple as tmdb
import time

tmdb.API_KEY = 'API_KEY'

path = '/home/seba/Uqido/movies-suggestions-bot/the-movies-dataset/'


movieId_n = 176219
counter = 0
years_aviable = [str(i) for i in range(2007, 2019)]

def get_tmdb_info(title, original_year):
    global movieId_n, years_aviable, counter
    search = tmdb.Search()
    response = search.movie(query=str(title))
    for res in search.results:
        year = res['release_date'].split('-')[0] 
        if str(year) == str(original_year):
            if float(res['popularity']) >= 7.0:
                print(title)
                movieId_n += 1

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

                links_csv = pd.read_csv(path + 'pop_new_links.csv')
                id_already_done = links_csv['tmdbId'].values
                for i in id_already_done:
                    if int(str(res['id'])) == int(i):
                        print("already in links")
                        return
                
                with open(path + 'pop_new_metadata.csv', 'a') as csvfile:
                    fieldnames = list(film.keys())
                    fieldnames.sort()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(film)

                with open(path + 'credits.csv', 'a') as csvfile:
                    fieldnames = ['cast','crew','id']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({'cast':cast, 'crew':crew, 'id':res['id']})
                
                with open(path + 'keywords.csv', 'a') as csvfile:
                    fieldnames = ['id', 'keywords']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({'id':res['id'], 'keywords':keywords['keywords']})
                
                with open(path + 'pop_new_links.csv', 'a') as csvfile:
                    fieldnames = ['useless', 'movieId', 'imdbId', 'tmdbId']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({'useless':0, 'movieId':movieId_n, 'imdbId':str(res['imdb_id'])[2:], 'tmdbId':str(res['id'])+'.0'})
                
                print("done")
                counter += 1
                time.sleep(0.5)
                return

url = "https://www.imdb.com/search/title?release_date=2007-01-01,2018-12-31&languages=en&sort=num_votes,desc&page="
for number_page in range(1, 100):
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
        if float(metascore/10) + imdb_rate < 14.0:
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
        md = pd.read_csv(path + 'pop_new_metadata.csv')
        title_already_done = md['title'].values
        if str(original_name) in title_already_done:
            print(original_name + "  already done")
            continue

        get_tmdb_info(original_name, year)
        

        # req_movie = requests.get(imdb_link)
        # data_movie = req_movie.text
        # data_movie = ''.join([i if ord(i) < 128 else '~' for i in data_movie])
        # soup_movie = BeautifulSoup(data_movie,"html.parser")
        # director = soup_movie.find('span', {'itemprop':"director", 'itemtype':"http://schema.org/Person"}).find('a').find('span').text
        # stars = []
        # for actors in soup_movie.findAll('span', {'itemprop':"actors", 'itemtype':"http://schema.org/Person"}):
        #     stars.append(str(actors.find('a').find('span').text))

        # req_keyword = requests.get(imdb_link + "/keywords")
        # data_keyword = req_keyword.text
        # data_keyword = ''.join([i if ord(i) < 128 else '~' for i in data_keyword])
        # soup_keyword = BeautifulSoup(data_keyword,"html.parser")


print(str(counter))