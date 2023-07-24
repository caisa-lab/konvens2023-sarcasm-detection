import tweepy
import re
from crawling.credentials import BEARER_TOKEN
from argparse import ArgumentParser
import time
import pandas as pd
import numpy as np
from credentials import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN_SECRET, ACCESS_TOKEN
from sarcasm_detection.util_functions import authenticate_api



# Example call:
# python get_non_sarcastic_thread.py --non_sarc_file=non_sarc.csv --buffer=non_sarc_thread_buffer.txt --output_file=non_sarc_thread.csv
parser = ArgumentParser(description='Fetches non-sarcastic thread')
parser.add_argument('--non_sarc_file', dest='non_sarc_file', help='Name of csv file that contains non-sarcastic tweets.', type=str)
parser.add_argument('--buffer', dest='buffer', help='Name of file that should be used as buffer. New file will be created.', type=str)
parser.add_argument('--output_file', dest= 'output_file', help='Name of file that should be used to save output csv. New csv file will be created.', type=str)

API = authenticate_api(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET, True)
CLIENT = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=False)
# search parameters
start_time = '2022-01-01T00:00:00Z'
end_time = '2022-12-01T00:00:00Z'
def get_single_tweet(tweet_id):
    """
    Method to crawl a single tweet object
    :param tweet_id: specific tweet id (str)
    :return:
    """
    sar_tweet = API.get_status(tweet_id)
    time.sleep(1)  # sleep because there is a request per second limit
    if sar_tweet is not None:
        tweet_text = re.sub(r'\s+', ' ', sar_tweet.text.replace('\n', ' '))
        re_user = sar_tweet.user.name + "|" + str(sar_tweet.user.id)

        return sar_tweet, str(sar_tweet.id), tweet_text, re_user, sar_tweet.in_reply_to_status_id_str

def get_eliciting_tweet(in_reply_to_status_id_str):
    """
    Gets the tweet to which the non-sarcastic tweets responds to, if non-sarcastic tweet is a reply
    Only call method if in_reply_to_status_id_str of non-sarcastic tweet != None
    :param in_reply_to_status_id_str:  id of parent tree if sarcastic tweet is reply
    :return:
    """
    eli_tweet = CLIENT.get_tweet(id=in_reply_to_status_id_str,
                                tweet_fields=['author_id,conversation_id,created_at,in_reply_to_user_id,referenced_tweets'],
                                user_fields=['id', 'username'],
                                expansions='author_id,in_reply_to_user_id,referenced_tweets.id,referenced_tweets.id.author_id'
                                )
    time.sleep(1)
    if eli_tweet is not None:
        tweet_text = re.sub(r'\s+', ' ', eli_tweet.data['text'].replace('\n', ' '))
        re_user = eli_tweet.includes['users'][0]['name'] + "|" + str(eli_tweet.data['author_id'])
        return str(eli_tweet.data['id']), tweet_text, re_user
    else:
        return None

def get_oblivious_tweet(tweet_id):
    """
    Gets the tweets, that the non-sarcastic tweet has been responding to
    :param tweet_id: tweet id of the non-sarcastic tweet (str)
    :return: list of lists for each reply containing the tweet id, tweet text, author id, author name
    """
    replies = []
    for tweet_ in tweepy.Paginator(CLIENT.search_all_tweets,
                                  query= 'in_reply_to_status_id:' + str(tweet_id),
                                  tweet_fields=['id,author_id,conversation_id,created_at,in_reply_to_user_id,referenced_tweets'],
                                  expansions='author_id,in_reply_to_user_id,referenced_tweets.id',
                                  start_time=start_time,
                                  end_time=end_time,
                                  max_results=10).flatten(10):
        time.sleep(1)  # sleep because there is a request per second limit
        text = re.sub(r'\s+', ' ', tweet_.text.replace('\n', ' '))
        replies.append([str(tweet_.id), text, str(tweet_.author_id)])
        for reply in replies: # get usernames of author
            author_id = reply[2]
            user = CLIENT.get_user(id=author_id)
            time.sleep(1)
            reply.append(user.data.name)
    return replies

def get_non_sarcastic_thread(non_sarc_file, buffer_file_name, output_file_name):
    buffer = []
    new_df_rows = []
    df_non_sarc = pd.read_csv(non_sarc_file)
    with open(buffer_file_name, 'a+') as outfile:
        for index, row in df_non_sarc.iterrows():
            try:
                obl_id, obl_text, obl_user, eli_id, eli_text, eli_user = None, None, None, None, None, None
                sar_tweet_object, sar_id, sar_text, sar_user, sar_reply = get_single_tweet(row['tweet_id'])
                obl_replies = get_oblivious_tweet(sar_id)
                if obl_replies: # oblivious replies not empty
                    obl_id = obl_replies[0][0]
                    obl_text = obl_replies[0][1]
                    obl_user = str(obl_replies[0][3]) + "|" + str(obl_replies[0][2])
                if sar_reply is not None:
                    eli_id, eli_text, eli_user = get_eliciting_tweet(sar_reply)

                buffer.append([sar_id, obl_id, eli_id, sar_text, obl_text, eli_text, sar_user, obl_user, eli_user])
                new_df_rows.append([sar_id, obl_id, eli_id, sar_text, obl_text, eli_text, sar_user, obl_user, eli_user])
                print('sar_id:{}, obl_id:{}, eli_id:{}, sar_text:{}, obl_text:{}, eli_text:{}, sar_user:{}, obl_user:{}, eli_user:{}'. format(sar_id, obl_id, eli_id, sar_text, obl_text, eli_text, sar_user, obl_user, eli_user))

                if len(buffer) >= 1000:
                    for data in buffer:
                        outfile.write(str(data[0]) + '\t' + str(data[1]) + '\t' + str(data[2]) + str(data[3]) + '\t' + str(data[4]) + '\t' + str(data[5]) + str(data[6]) + '\t' + str(data[7]) + '\t' + str(data[8]) + '\n')
                    buffer.clear()

            except tweepy.HTTPException as e:
                if 401 in e.api_codes or e.response.status_code == 401:
                    print('encountered HTTP error 401 "Unauthorized" for tweet id {}'.format(row['tweet_id']))
                    continue  # continue the loop with other tweet_id
                elif 404 in e.api_codes or e.response.status_code == 404:
                    print('encountered HTTP error 404 "User not found" for tweet id {}'.format(row['tweet_id']))
                    continue  # continue the loop with other tweet_id
                elif 429 or 503 in e.api_codes or e.response.status_code == 429 or e.response.status_code == 503:
                    print('encountered HTTP error 429/503 "To many requests" for tweet id {}. Sleeping for 900 seconds.'.format(row['tweet_id']))
                    time.sleep(15 * 60)
                else:
                    print('The following Exception occurred {}'.format(e))
                    print('Last tweet id: {}'.format(row['tweet_id']))
            except Exception as es:
                print('The following Exception occurred {}'.format(es))
                print('Last tweet id: {}'.format(row['tweet_id']))

        # new_df_rows
        df = pd.DataFrame(np.asarray(new_df_rows))  # create df out of matrix
        df = df.rename(columns={0: "sar_id", 1: "obl_id", 2: "eli_id",3: "sar_text", 4: "obl_text", 5: "eli_text", 6: "sar_user", 7: "obl_user", 8: "eli_user"})
        df.to_csv(output_file_name, index=False)
        print("Successfully crawled cue tweets")


if __name__ == '__main__':
    args = parser.parse_args()
    get_non_sarcastic_thread(args.non_sarc_file, args.buffer, args.output_file)



