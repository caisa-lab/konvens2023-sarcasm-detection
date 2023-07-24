#!/usr/bin/env python

import tweepy
import pandas as pd
import numpy as np
import argparse
import time
from credentials import BEARER_TOKEN
from sarcasm_detection.util_functions import get_current_timestamp

TIMESTAMP = get_current_timestamp()
# define client (change bearer token)
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

# define components to be parsed -> example how to open the file python fetch_cue_tweets_client.py "data_buffer.txt"
# "fetched_cue_tweets.csv" "raw_cue_tweets.csv" 10
parser = argparse.ArgumentParser(description='Fetches cue tweets containing "being sarcastic"')
parser.add_argument('out_path', help='tab-separated file consisting of [tweet_id, tweet_text, user id] lists.', type=str)
parser.add_argument('output_file', help='CSV file consisting of [tweet_id, tweet_text, user id] lists.', type=str)
parser.add_argument('raw_data', help='CSV file consisting of raw data of api search.', type=str)
parser.add_argument('max_tweets', type=int, default=100, help='Max number of cue tweets to be fetched')



# define parameters for search (time period to search in)
# start period
start_time = '2022-01-01T00:00:00Z'
# end period
end_time = '2022-11-01T00:00:00Z'


# method to fetch cue tweets
def fetch_cue_tweets(out_path, output_file, raw_data, max_tweets):
    """
    Saves cue tweets containing 'being sarcastic' for numer of max_tweets.
    Uses search_all_tweets see: https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-all
    :param out_path: name of output buffer file (str)
    :param output_file: name of output csv file (str)
    :param raw_data: name of output raw csv file (str)
    :param max_tweets: maximum tweets to be crawled (int)
    :return:
    """
    data_buffer = list()  # buffer for saving results in case of exceptions
    with open(out_path, 'a+') as outfile:
        try:
            # search for query in specified time period
            new_rows = []  # for matrix of tweets
            searched_tweets = []  # for row data
            for tweet in tweepy.Paginator(client.search_all_tweets,
                                          query='being sarcastic -is:retweet lang:en',
                                          user_fields=['id', 'username'],
                                          expansions='author_id',
                                          start_time=start_time,
                                          end_time=end_time,
                                          max_results=500).flatten(limit=max_tweets):
                time.sleep(1)  # sleep because there is a request per second limit
                print('tweet_id: {}, tweet_text: {}, tweet_author_id: {}'.format(tweet.id, tweet.text, tweet.author_id))
                searched_tweets.append(tweet)
                new_rows.append([tweet.id, tweet.text, tweet.author_id])
                data_buffer.append([tweet.id, tweet.text, tweet.author_id])

                # save buffer
                if len(data_buffer) >= 1000:
                    for data in data_buffer:
                        outfile.write(str(data[0]) + '\t' + str(data[1]) + '\t' + str(data[2]) + '\n')
                    data_buffer.clear()

            # save data to csv
            df = pd.DataFrame(np.asarray(new_rows))  # create df out of matrix
            df = df.rename(columns={0: "tweet_id", 1: "text", 2: "user_id"})
            df.to_csv(output_file, index=False)
            pd.DataFrame(np.asarray(searched_tweets)).to_csv(raw_data, index=False)
            print("Successfully crawled {} cue tweets".format(max_tweets))

        except Exception as e:
            print(e)

        finally:
            for data in data_buffer:
                outfile.write(str(data[0]) + '\t' + str(data[1]) + '\t' + str(data[2]) + '\n')
            data_buffer.clear()
            print("Writing to data")


if __name__ == '__main__':
    # parse input
    args = parser.parse_args()
    # method call
    fetch_cue_tweets(args.out_path, args.output_file, args.raw_data, args.max_tweets)
