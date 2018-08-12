import logging
import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, Filters, MessageHandler, InlineQueryHandler
from telegram.ext.dispatcher import run_async

import numpy as np
import copy, time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from read_csv import get_titles, get_most_poular, add_rating, add_user, user_gender, get_row_title
from hybrid import final_res, movies_from_last_one, list_movies_seen_user, list_movies_seen_user, predict_one_movie
from similar_user import get_similar_user
from movies_cinema_imdb import final_cinema_movies, get_all_cinema_movies


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

telegram.constants.MAX_MESSAGE_LENGTH = 10000
telegram.constants.MAX_CAPTION_LENGTH = 1000

flag_input_start = False

# title deve essere in lowercase
# partial se vuole tutti i possibili risulati di quel tiolo, altrimenti resituisce un nome solo (se lo trova)
def search(title, partial=True):
    possible_original_title = []
    titles_movies = get_titles()
    for t in titles_movies:
        t_copy = copy.deepcopy(t).lower()
        if title in t_copy:
            possible_original_title.append(t)

    if partial:
        return possible_original_title

    if possible_original_title:
        # prendo quello che ha la lunghezza piu' vicina a "title"
        original_title = possible_original_title[0]
        for t in possible_original_title:
            if abs(len(title)-len(t)) < abs(len(title)-len(original_title)):
                original_title = t
        return original_title
    return None


# creo l'interfaccia per tutti i possibili titoli
def input_movie_user(bot, update, possible_original_title, rate=0, info=False):
    keyboard = [[i] for i in range(len(possible_original_title))]
    count = 0
    if info:
        for title in possible_original_title:
            keyboard[count][0] = InlineKeyboardButton(title, callback_data="5 " + title)
            count += 1
    elif rate == 0:
        for title in possible_original_title:
            keyboard[count][0] = InlineKeyboardButton(str(title), callback_data="3 " + str(title))
            count += 1
    else:
        for title in possible_original_title:
            keyboard[count][0] = InlineKeyboardButton(title, callback_data="4 " + title + " || " + str(rate))
            count += 1
    try:
        reply_markup = InlineKeyboardMarkup(keyboard)
        if len(possible_original_title) == 1:
            update.message.reply_text('Is This?', reply_markup=reply_markup)
        else:
            update.message.reply_text('Which one?', reply_markup=reply_markup)
    except:
        chat_id = update.message.chat_id
        bot.send_message(chat_id=chat_id, text="Invalid name \n[Too short/long, or does't exist]")
        bot.send_message(chat_id, 'Insert the name of the movie', reply_markup=telegram.ForceReply(force_reply=True))
        flag_input_start = True


@run_async
def start(bot, update):
    movies = get_most_poular()
    chat_id = update.message.chat_id

    name = str(update['message']['chat']['username'])
    if len(name) < 5:
        bot.send_message(chat_id=chat_id, text="\t-- !ERROR! --\nYour data won't be save\nYou must have a username set on Telegram\n"\
                        "[How to do it](https://telegram.org/blog/usernames-and-secret-chats-v2)", parse_mode=telegram.ParseMode.MARKDOWN)
        return

    kb = [
          [telegram.KeyboardButton('/rating'), telegram.KeyboardButton('/movies')],
          [telegram.KeyboardButton('/rate'), telegram.KeyboardButton('/search'), telegram.KeyboardButton('/list')],
          [telegram.KeyboardButton('/friend'), telegram.KeyboardButton('/cinema')]
         ]
    kb_markup = telegram.ReplyKeyboardMarkup(kb, resize_keyboard=True)
    
    # keyboard = [[i] for i in range(len(movies)+1)]
    # keyboard[0][0] = InlineKeyboardButton("Write the name manually", callback_data="2 ")
    keyboard = [[i] for i in range(len(movies))]
    count = 0
    for title, img in movies:
        keyboard[count][0] = InlineKeyboardButton(title, callback_data="3 " + title)
        count += 1

    reply_markup = InlineKeyboardMarkup(keyboard)

    if list_movies_seen_user(name, True) > 0:
        update.message.reply_text('Hi @' + name + "\nYou have already started the adventure ;)\n/rating to add feedback, or /movies to get your recommended movies", 
                                        reply_markup=kb_markup)
    else:
        bot.send_message(chat_id=chat_id, text="Hi @" + name, reply_markup=kb_markup)
        msg = "Here there is a quick review of the available commands:\n\n" +\
            "rate - Rate a movie { /rate Interstellar 5 }\n" +\
            "search - Get info about a movie { /search Interstellar }\n" +\
            "list - Get the list of movies rated by you { /list }\n" +\
            "rating - Add rates to movies { /rating }\n" +\
            "movies - Recommended movies { /movies }\n" +\
            "friend - Find a person similar to you { /friend }\n" +\
            "cinema - Recommended movies at the cinema now { /cinema }"
        bot.send_message(chat_id=chat_id, text=msg, reply_markup=kb_markup)
        update.message.reply_text("Let begin to know each other ;)\nChoose or search (see /rate) your favorite movie", reply_markup=reply_markup)


# def contact(bot, update):
#     contact_keyboard = telegram.KeyboardButton(text="send_contact", request_contact=True)
#     custom_keyboard = [[contact_keyboard]]
#     reply_markup = telegram.ReplyKeyboardMarkup(custom_keyboard)
#     chat_id = update.message.chat_id
#     bot.send_message(chat_id=chat_id, 
#                      text="Would you mind sharing your contact with me?", 
#                      reply_markup=reply_markup)

@run_async
def button(bot, update):
    query = update.callback_query
    chat_id = query.message.chat_id
    option = str(query.data.split(' ')[0])

    # # INPUT USER
    # if option == "2":
    #     global flag_input_start
    #     bot.send_message(chat_id, 'Insert the name of the movie', reply_markup=telegram.ForceReply(force_reply=True))
    #     flag_input_start = Truecheck_user_exist

    # SHOW MOVIES AND RATING
    if option == "3":
        title = ' '.join(query.data.split(' ')[1:])
        user_name = str(update['callback_query']['from_user']['username'])

        bot.edit_message_text(text="Selected option: {0}".format(title),
                              chat_id=chat_id, message_id=query.message.message_id)

        keyboard = [
                     [
                        InlineKeyboardButton("1", callback_data="4 " + title + " || 1"),
                        InlineKeyboardButton("2", callback_data="4 " + title + " || 2"), 
                        InlineKeyboardButton("3", callback_data="4 " + title + " || 3")
                     ],
                     [
                        InlineKeyboardButton("4", callback_data="4 " + title + " || 4"), 
                        InlineKeyboardButton("5", callback_data="4 " + title + " || 5")
                     ]
                   ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        bot.send_message(chat_id=chat_id, text='Insert your personal rating for the movie', reply_markup=reply_markup)
    
    # SAVE AND ASK MOVIE AGAIN
    elif option == "4":
        option2 = ' '.join(query.data.split(' ')[1:])
        user_name = str(update['callback_query']['from_user']['username'])
        title, rating = option2.split(' || ')

        bot.edit_message_text(text="{0}/5 for {1}".format(int(rating), title),
                              chat_id=chat_id, message_id=query.message.message_id)

        add_rating(user_name, title, int(rating))

        recommended_movies = movies_from_last_one(str(user_name))

        keyboard = [[i] for i in range(len(recommended_movies))]
        count = 0
        for title, value in recommended_movies:
            keyboard[count][0] = InlineKeyboardButton(str(title), callback_data="3 " + str(title))
            count += 1

        reply_markup = InlineKeyboardMarkup(keyboard)

        n_movies = list_movies_seen_user(user_name, True)

        if n_movies == 1:
            bot.send_message(chat_id=chat_id, text='Nice choice!\nNow what', reply_markup=reply_markup)
        elif n_movies == 2:
            bot.send_message(chat_id=chat_id, text='Ok then one more?', reply_markup=reply_markup)
        elif n_movies == 3:
            msg = "Good job! Now you are ready to continue your adventure without worries\n" +\
                   "When you think to be ready do /movies to get your recommended movies\n"+ \
                   'Choose a movie to rate'
            bot.send_message(chat_id=chat_id, text=msg, reply_markup=reply_markup)
        elif n_movies == 5:
            msg = "Feel free to send me some feedback about the bot [link](https://goo.gl/forms/02lDSAURiQq2Cnzg1)"
            bot.send_message(chat_id=chat_id, text=msg, reply_markup=reply_markup, parse_mode=telegram.ParseMode.MARKDOWN)
            bot.send_message(chat_id=chat_id, text='Choose a movie to rate', reply_markup=reply_markup)
        else:
            bot.send_message(chat_id=chat_id, text='Choose a movie to rate', reply_markup=reply_markup)

    # INFO MOVIE
    elif option == "5":
        user_name = str(update['callback_query']['from_user']['username'])
        title = ' '.join(query.data.split(' ')[1:])

        row_title = get_row_title(title) 

        predict = None
        overview = None
        try:
            overview = row_title['overview'].values[0]
        except:
            overview = ""

        try:
            predict = predict_one_movie(user_name, int(row_title['id'].values[0]))
        except:
            predict = "No info"

        message = "*" + str(row_title['title'].values[0]).upper() + "* \n" + \
                  "*Year Release*: " + str(int(row_title['year'].values[0])) + "\n" + \
                  "*Genres*: " + ','.join([str(i) for i in row_title['genres']]) + "\n" + \
                  "*Runtime*: " + str(int(row_title['runtime'].values[0])) + " minutes\n" + \
                  "*Possible Rate*: " + str(predict) + "\n" +\
                  "*Overview*:\n" + str(overview)
        # "[How to do it](https://telegram.org/blog/usernames-and-secret-chats-v2)"                

        keyboard = [[InlineKeyboardButton("Trailer", url="https://www.youtube.com/results?search_query=" + str(title) + " trailer")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        bot.edit_message_text(chat_id=chat_id, text=message, message_id=query.message.message_id, parse_mode=telegram.ParseMode.MARKDOWN, reply_markup=reply_markup)
        bot.send_photo(chat_id=chat_id, photo="https://image.tmdb.org/t/p/original/" + str(row_title['poster_path'].values[0]))

    # YES or NO for /friend + ask GENDER
    elif option == "6":
        answer = query.data.split(' ')[1]
        user_name = str(update['callback_query']['from_user']['username'])
        if answer == "yes":
            bot.edit_message_text(chat_id=chat_id, text="Welcome to this amazing feature :D\n", message_id=query.message.message_id)
            keyboard = [[InlineKeyboardButton("Male", callback_data="7 male"), InlineKeyboardButton("Female", callback_data="7 female")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            bot.send_message(chat_id=chat_id, text='Now I need to know your gender and your age\nAre you a boy or a girl?', reply_markup=reply_markup)
        else:
            bot.edit_message_text(chat_id=chat_id, text="I can't continue without your agreement :(", message_id=query.message.message_id)

    # GENDER + ask AGE
    elif option == "7":
        answer = query.data.split(' ')[1]
        user_name = str(update['callback_query']['from_user']['username'])

        keyboard = [
                    [InlineKeyboardButton("1-10", callback_data="8 " + answer + " 1"), InlineKeyboardButton("11-20", callback_data="8 " + answer + " 2"),
                        InlineKeyboardButton("21-30", callback_data="8 " + answer + " 3")],
                    [InlineKeyboardButton("31-40", callback_data="8 " + answer + " 4"), InlineKeyboardButton("41-50", callback_data="8 " + answer + " 5"),
                        InlineKeyboardButton("50+", callback_data="8 " + answer + " 6")]
                   ]
        reply_markup = InlineKeyboardMarkup(keyboard)        
        bot.send_message(chat_id=chat_id, text='And what about your age range?', reply_markup=reply_markup)
    
    # SAVE USER
    elif option == "8":
        user_name = str(update['callback_query']['from_user']['username'])
        gender = query.data.split(' ')[1]
        age = query.data.split(' ')[2]
        
        add_user(user_name, gender, age)

        bot.send_message(chat_id=chat_id, text='Nice to meet you :)')

        similar_user, percent_similarity = get_similar_user(user_name)
        if similar_user == "":
            bot.send_message(chat_id=chat_id, text='No user found :(')
        else:
            gender_user = user_gender(user_name)
            if gender_user != user_gender(similar_user):
                bot.send_message(chat_id=chat_id, text='WooW! Soo lucky ;)\n@' + similar_user + " with a similarity of " + percent_similarity + "%")
            else:
                bot.send_message(chat_id=chat_id, text="Here it is a person with tastes similar to yours\n@" + similar_user +\
                                    " with a similarity of " + percent_similarity + "%")
        
    # LIST MOVIES AT CINEMA + RECOMMENDED CINEMA MOVIES
    elif option == "9":
        user_name = str(update['callback_query']['from_user']['username'])
        answer = query.data.split(' ')[1]
        if answer == "1":
            all_cinema_movies = get_all_cinema_movies()

            keyboard = [[i] for i in range(len(all_cinema_movies))]
            count = 0
            for title, available in all_cinema_movies:
                if available == "":
                    keyboard[count][0] = InlineKeyboardButton(str(title), callback_data="5 " + str(title))
                else:
                    keyboard[count][0] = InlineKeyboardButton(str(title) + " [" + str(available) + "]", callback_data="5 " + str(title))
                count += 1

            reply_markup = InlineKeyboardMarkup(keyboard)
            bot.edit_message_text(chat_id=chat_id, text='Best movies at cinema for you ;)\nClick one to have more info', 
                                reply_markup=reply_markup, message_id=query.message.message_id)

        else:
            bot.edit_message_text(chat_id=chat_id, text="Have you already searched for the nearest cinema? ;)\nPlease wait some seconds", message_id=query.message.message_id)
            recommended_movies = final_cinema_movies(user_name)

            keyboard = [[i] for i in range(len(recommended_movies))]
            count = 0
            for title, rate in recommended_movies:
                keyboard[count][0] = InlineKeyboardButton(str(title), callback_data="5 " + str(title))
                count += 1

            reply_markup = InlineKeyboardMarkup(keyboard)
            bot.send_message(chat_id=chat_id, text='Here it is your recommanded movies :)\nHave fun at cinema ;)\nClick one to have more info', reply_markup=reply_markup)

        
# @run_async        
# def input_user(bot, update):
#     global flag_input_start
#     chat_id = update.message.chat_id

#     if flag_input_start == True:
#         flag_input_start = False
#         title = str(update.message.text).lower()
#         user_name = str(update['message']['chat']['username'])

#         possible_original_title = search(title)

#         if possible_original_title:
#             input_movie_user(bot, update, possible_original_title)
#         else:
#             bot.send_message(chat_id=chat_id, text="Invalid name \n[Too short/long, or does't exist]")

@run_async
def rate_movie(bot, update, args):
    chat_id = update.message.chat_id
    if len(args) < 1:
        bot.send_message(chat_id=chat_id, text="Invalid argument: /rate name [rate]\nExample:\n\t/rate Interstellar 5")
    else:
        title, rate = '', ''
        if args[-1].isdigit():
            if not int(args[-1]) in [1, 2, 3, 4, 5]:
                bot.send_message(chat_id=chat_id, text="Invalid rate \n[Must be 1,2,3,4 or 5]")
                return 
            title, rate = ' '.join(args[:-1]), args[-1]
        else:
            title = ' '.join(args)

        possible_original_title = search(title.lower())
        if len(possible_original_title):
            if rate:
                input_movie_user(bot, update, possible_original_title, rate)
            else:
                input_movie_user(bot, update, possible_original_title)
        else:
            bot.send_message(chat_id=chat_id, text="Invalid name \n[Too short/long, or does't exist]")

@run_async
def search_movie(bot, update, args):
    chat_id = update.message.chat_id
    if len(args) < 1:
        bot.send_message(chat_id=chat_id, text="Invalid argument: /search name\nExample:\n\t/search Interstellar")
    else:
        title = ' '.join(args)
        possible_original_title = search(title.lower())
        if len(possible_original_title):
            input_movie_user(bot, update, possible_original_title, info=True)
        else:
            bot.send_message(chat_id=chat_id, text="Invalid name \n[Too short/long, or does't exist]")

@run_async
def rating_movies(bot, update):
    user_name = str(update['message']['chat']['username'])
    chat_id = update.message.chat_id

    recommended_movies = movies_from_last_one(user_name)

    if len(recommended_movies) < 1:
        bot.send_message(chat_id=chat_id, text='You should rate same movie befor start.\n/start to start')
        return 

    keyboard = [[i] for i in range(len(recommended_movies))]
    count = 0
    for title, value in recommended_movies:
        keyboard[count][0] = InlineKeyboardButton(str(title), callback_data="3 " + str(title))
        count += 1

    reply_markup = InlineKeyboardMarkup(keyboard)
    bot.send_message(chat_id=chat_id, text='Choose a movie to rate', reply_markup=reply_markup)


@run_async
def get_best_movie(bot, update):
    user_name = str(update['message']['chat']['username'])
    chat_id = update.message.chat_id

    bot.send_message(chat_id=chat_id, text="Remember to do /rating to continue the adventure!\nPlease wait some seconds for the recommendetion")

    recommended_movies = final_res(user_name)

    if len(recommended_movies) < 1:
        bot.send_message(chat_id=chat_id, text='You should rate same movie befor start.\n/start to start')
        return 

    keyboard = [[i] for i in range(len(recommended_movies))]
    count = 0
    for title, rate in recommended_movies:
        keyboard[count][0] = InlineKeyboardButton(str(title) + " -> " + str(rate)[:5], callback_data="5 " + str(title))
        count += 1

    reply_markup = InlineKeyboardMarkup(keyboard)
    bot.send_message(chat_id=chat_id, text='Here it is your recommanded movies :)\nClick one to have more info', reply_markup=reply_markup)

@run_async
def get_list_movies(bot, update):
    user_name = str(update['message']['chat']['username'])
    chat_id = update.message.chat_id

    lm = list_movies_seen_user(user_name)
    if len(lm) == 0:
        bot.send_message(chat_id=chat_id, text='You should rate same movie befor start.\n/start to start')
    else:
        bot.send_message(chat_id=chat_id, text='\n'.join(lm))

@run_async
def find_friend(bot, update):
    user_name = str(update['message']['chat']['username'])
    chat_id = update.message.chat_id

    if list_movies_seen_user(user_name, True) < 10:
        bot.send_message(chat_id=chat_id, text='You have to rate at least 10 movies before get into this section\n/rating to continue')
        return

    gender_user = user_gender(user_name)
    if not gender_user:
        bot.send_message(chat_id=chat_id, text='In this section you can find a friend ( or maybe something else ;) ) to watch a movie with')
        
        keyboard = [[InlineKeyboardButton("Yes", callback_data="6 yes"), InlineKeyboardButton("No", callback_data="6 no")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        bot.send_message(chat_id=chat_id, text='But first you have to agree to share your personal information with other people\nDo you agree?', 
                        reply_markup=reply_markup)
    else:

        similar_user, percent_similarity = get_similar_user(user_name)
        if similar_user == "":
            bot.send_message(chat_id=chat_id, text='No user found :(')
        else:
            if gender_user != user_gender(similar_user):
                bot.send_message(chat_id=chat_id, text='WooW! Soo lucky ;)\n@' + similar_user + " with a similarity of " + percent_similarity + "%")
            else:
                bot.send_message(chat_id=chat_id, text="Here it is a person with tastes similar to yours\n@" + similar_user +\
                                    " with a similarity of " + percent_similarity + "%")

@run_async
def movies_cinema(bot, update):
    user_name = str(update['message']['chat']['username'])
    chat_id = update.message.chat_id

    lm = list_movies_seen_user(user_name)
    if len(lm) == 0:
        bot.send_message(chat_id=chat_id, text='You should rate same movie befor start.\n/start to start')
    else:
        msg = "Choose an option:\n1) Show the most popular movies at cinema now\n" +\
                "2) A recommendetion of the best movies, based on your preference, to watch at cinema now!"
        keyboard = [[InlineKeyboardButton("1", callback_data="9 1"), InlineKeyboardButton("2", callback_data="9 2")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        bot.send_message(chat_id=chat_id, text=msg, reply_markup=reply_markup)

        

# @run_async
# def inline_movies(bot, update):
#     user_name = str(update['inline_query']['from_user']['username'])
#     chat_id = update.inline_query.from_user.id

#     query = update.inline_query.query
#     results = list()

#     if query == 'movies' or query == "Movies":
#         pass

#         if not str(user_name) in list_name_users:
#             bot.send_message(chat_id=chat_id, text='You should rate same movie befor start.\n/start to start')
#             return 

#         recommended_movies = final_res(str(user_name))
#         # return_msg = ""
#         # for key, value in recommended_movies:
#         #     return_msg += str(key) + " -> " + str(value)[5:] + "\n" 
#         keyboard = [[i] for i in range(len(recommended_movies))]
#         count = 0
#         for key, value in recommended_movies:
#             keyboard[count][0] = InlineKeyboardButton(str(key) + " -> " + str(value)[:5], callback_data="3 " + key)
#             count += 1

#         reply_markup = InlineKeyboardMarkup(keyboard)
#         bot.send_message(chat_id=chat_id, text='Here it is your recommanded movies :)', reply_markup=reply_markup)

#         results.append(
#             InlineQueryResultArticle(
#                 id=chat_id,
#                 title='Movies',
#                 input_message_content=InputTextMessageContent('Choose a movie to rate'),
#                 reply_markup=reply_markup
#             )
#         )
#     elif query == 'search':
#         return
#     elif query == 'rate':
#         return
#     else:
#         return
    
#     bot.answer_inline_query(update.inline_query.id, results)


from telegram.error import (TelegramError, Unauthorized, BadRequest, 
                            TimedOut, ChatMigrated, NetworkError)
def error_callback(bot, update, error):
    logger.warning('Update "%s" caused error "%s"', update, error)
    try:
        raise error
    except Unauthorized:
        print("Unauthorized")
        # remove update.message.chat_id from conversation list
    except BadRequest as e:
        print("BadRequest")
        print(e)
        # handle malformed requests - read more below!
    except TimedOut:
        print("TimedOut")
        # handle slow connection problems
    except NetworkError:
        print("NetworkError")
        # handle other connection problems
    except ChatMigrated as e:
        print("ChatMigrated")
        print(e)
        # the chat_id of a group has changed, use e.new_chat_id instead
    except TelegramError:
        print("TelegramError")
        # handle all other telegram related errors


ud = updater.dispatcher
ud.add_handler(CommandHandler('start', start))
ud.add_handler(CommandHandler('rating', rating_movies))
ud.add_handler(CommandHandler('movies', get_best_movie))
ud.add_handler(CommandHandler('list', get_list_movies))
ud.add_handler(CommandHandler('friend', find_friend))
ud.add_handler(CommandHandler('cinema', movies_cinema))
ud.add_handler(CommandHandler('rate', rate_movie, pass_args=True))
ud.add_handler(CommandHandler('search', search_movie, pass_args=True))
ud.add_handler(CallbackQueryHandler(button)) # , pattern='main'
# ud.add_handler(CallbackQueryHandler(info_film))

# ud.add_handler(InlineQueryHandler(inline_movies))

# ud.add_handler(MessageHandler(Filters.text, input_user))
ud.add_error_handler(error_callback)

updater.start_polling(timeout=100.0)
updater.idle()