import tweepy
import re
from crawling.credentials import BEARER_TOKEN
from argparse import ArgumentParser
import time
import pandas as pd
import numpy as np

# authentication to use client functions like search_all
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)
# define parameters for search (time period to search in)
start_time = '2022-01-01T00:00:00Z'  # start period
end_time = '2022-11-01T00:00:00Z'  # end period
QUERY = '-sarcastic -sarcasm -#sarcasticquotes -#sarcasticquote -#sarcasticmemes -#sarcastic -#sarcasm -is:retweet lang:en'

# Example call:
# python fetch_non_sarcastic_tweets.py --buffer=buffer.txt --output_file='non_sarc.csv' --max_tweets=20000
parser = ArgumentParser(description='Fetches non-sarcastic tweets')
parser.add_argument('--buffer', dest='buffer', help='Name of file that should be used as buffer. New file will be created.', type=str)
parser.add_argument('--output_file', dest= 'output_file', help='Name of file that should be used to save output csv. New csv file will be created.', type=str)
parser.add_argument('--max_tweets',dest='max_tweets', type=int, default=100, help='Max number of non sarcastic tweets to be fetched')

def fetch_non_sarcastic_tweets(buffer_file, output_file, max_tweets):
    """
    Method that crawls random non-sarcastic related tweets.
    :param buffer_file: Name of file that should be used as buffer. New file will be created (str).
    :param output_file: Name of file that should be used to save output csv. New csv file will be created (str).
    :param max_tweets: Max number of non-sarcastic tweets to be fetched (int).
    :return: saves data to csv and buffer file
    """

    data_buffer = list()  # buffer for saving results in case of exceptions
    df_matrix = []  # for matrix of tweets
    print('Crawling {} random tweets '.format(max_tweets))
    with open(buffer_file, 'a+') as outfile:
        try:
            for tweet_ in tweepy.Paginator(client.search_all_tweets,
                                           query=QUERY,
                                           user_fields=['id', 'username'],
                                           expansions='author_id',
                                           start_time=start_time,
                                           end_time=end_time,
                                           max_results=500).flatten(limit=max_tweets):
                time.sleep(1.0) # sleep because there is a request per second limit
                text = re.sub(r'\s+', ' ', tweet_.text.replace('\n', ' '))
                print('tweet_id: {}, tweet_text: {}, tweet_author_id: {}'.format(tweet_.id, text, tweet_.author_id))
                df_matrix.append([tweet_.id, text, tweet_.author_id])
                data_buffer.append([tweet_.id, text, tweet_.author_id])

                # save buffer into file
                if len(data_buffer) >= 1000:
                    for data in data_buffer:
                        outfile.write(str(data[0]) + '\t' + str(data[1]) + '\t' + str(data[2]) + '\n')
                        data_buffer.clear()

            # save everything to csv
            df = pd.DataFrame(np.asarray(df_matrix))  # create df out of matrix
            df = df.rename(columns={0: "tweet_id", 1: "text", 2: "user_id"})
            df.to_csv(output_file, index=False)
            print("Successfully crawled cue tweets")

        except tweepy.HTTPException as e:
            if 429 or 503 in e.api_codes or e.response.status_code == 429 or e.response.status_code == 503:
                print('encountered HTTP error 429/503 "To many requests"')
                time.sleep(15 * 60)
            else:
                print('The following Exception occurred {}'.format(e))


if __name__ == '__main__':
    args = parser.parse_args()
    fetch_non_sarcastic_tweets(args.buffer, args.output_file, args.max_tweets)




