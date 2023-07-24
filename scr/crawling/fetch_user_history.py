import tweepy
import re
import pandas as pd
from credentials import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN_SECRET, ACCESS_TOKEN
from sarcasm_detection.util_functions import authenticate_api
from argparse import ArgumentParser
import os
import time


# first authenticate api
API = authenticate_api(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET, True)

# Example call
# python fetch_user_history.py --user_csv=mapped_cue_tweets_subset.csv --user_column="sar_user" --buffer_name=user_hist_buffer.txt --result_file_name=user_hist.txt
parser = ArgumentParser()
parser.add_argument("--user_csv", dest="user_csv", type=str, help="File containing user ids you want to crawl the user history for.")
parser.add_argument("--user_column", dest="user_column", type=str, help="Name of the column in which the user id is.")
parser.add_argument("--buffer_name", dest="buffer_name", type=str, help="File name for buffer file.")
parser.add_argument("--result_file_name", dest="result_file_name", type=str, help="File name for resulting file.")

def lookup_author(user, count=200):
    """ unction to look up the user timeline using the user id of a specific user
    see  https://developer.twitter.com/en/docs/twitter-api/v1/tweets/timelines/api-reference/get-statuses-user_timeline
    count can be maximum=200; count applies to each request made by Cursor
    Cursor can request up to 3200 tweets
    :param user: twitter user id (str)
    :param count: number between 1 - 200 (int)
    :return: user timeline iterator
    """
    for stat in tweepy.Cursor(API.user_timeline, user_id=user, tweet_mode='extended', count=count).items():
        yield stat

def get_user_hist(user_csv, user_column, buffer_name, result_file):
    """
    Creates a user history file for all users provided in user_csv
    :param user_csv: csv file containing users (str)
    :param user_column: name of column in which users are saved (str)
    :param buffer_name: file name for saving into a buffer file (str)
    :param result_file: name of resulting txt file (str)
    :return: No return, only saves internally
    """
    buffer = list()
    buffer_file = os.path.join("", buffer_name) # change the directory as you like
    out_list = list()
    re_file = os.path.join("", result_file) # change the directory as you like
    df_user = pd.read_csv(user_csv)
    with open(buffer_file, 'a+') as outfile:
        for i, row in df_user.iterrows():
            user_id = row[user_column].split("|")
            user_id = user_id[1]
            try:
                for tweet in lookup_author(user_id):
                    tweet_id = getattr(tweet, 'id')
                    tweet_text = re.sub(r'\s+', ' ', getattr(tweet, 'full_text', '').replace('\n', ' '))
                    created_at = getattr(tweet, 'created_at')
                    print("User id {}, Tweet id {}, Tweet text {}".format(user_id,tweet_id,tweet_text))
                    buffer.append([user_id, tweet_id, tweet_text, created_at]) # save in buffer
                    out_list.append([user_id, tweet_id, tweet_text, created_at])
                    with open(re_file, 'a+') as result: # write information to file
                        result.write(str(user_id) + '\t' + str(tweet_id) + '\t' + str(tweet_text) + '\t' + str(created_at) + '\n')

                    if len(buffer) >= 1000: # make savings in case of an exception
                        for data in buffer:
                            outfile.write(str(data[0]) + '\t' + str(data[1]) + '\t' + str(data[2]) + '\t' + str(data[3]) + '\n')
                        buffer.clear()
            except tweepy.HTTPException as e:
                if 401 in e.api_codes or e.response.status_code == 401:
                    print('encountered HTTP error 401 "Unauthorized" for user {}'.format(user_id))
                    continue # continue the loop with other user (skip the user)
                elif 404 in e.api_codes or e.response.status_code == 404:
                    print('encountered HTTP error 404 "User not found" for user {}'.format(user_id))
                    continue # continue the loop with other user (skip the user)
                elif 429 or 503 in e.api_codes or e.response.status_code == 429 or e.response.status_code == 503:
                    print('encountered HTTP error 429/503 "To many requests" for user {}'.format(user_id))
                    time.sleep(15*60)
            except Exception as es:
                print('encountered the following error {} for the user {}'.format(es,user_id))
                print('Sleeping for 900 seconds.')
                time.sleep(15 * 60)
            finally:
                print('Last user id:{}'.format(user_id))
                outfile.write(str(user_id))
    print('User history has been successfully saved for {} out of {} users.'.format(str(len(out_list)), str(len(df_user))))


if __name__ == '__main__':
    args = parser.parse_args()
    # call method
    get_user_hist(args.user_csv, args.user_column, args.buffer_name, args.result_file_name)