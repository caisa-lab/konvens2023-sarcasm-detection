#!/usr/bin/env python
import argparse
import sys
import pandas as pd
import re
import time
import tweepy
from credentials import BEARER_TOKEN

# authentication to use client functions like search_all
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# example call of script: python get_sarcastic_thread.py "buffer_map.txt" "filtered_cue_tweets_client"
# "mapped_cue_tweets.csv"
parser = argparse.ArgumentParser(description='maps cue tweets')
parser.add_argument('buffer_txt', help='tab-separated file consisting of [tweet_id, tweet threads] lists.', type=str)
parser.add_argument('input_csv', help='CSV file consisting of [tweet_id, tweet_text, user id] lists.', type=str)
parser.add_argument('output_csv', help='CSV file consisting of raw data of api search.', type=str)


# To generate next letter (capitalised) for the identification of tweet autors in thread
def next_alpha(s):
    return chr((ord(s.upper()) + 1 - 65) % 26 + 65)


def get_single_tweet(tweet_id):
    while True:
        try:
            tweet = client.get_tweet(id=tweet_id,
                                     tweet_fields=['author_id,conversation_id,created_at,in_reply_to_user_id,referenced_tweets'],
                                     user_fields=['id', 'username'],
                                     expansions='author_id,in_reply_to_user_id,referenced_tweets.id'
                                     )
            return tweet
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print('The following exception occurred: {}. Wait 60*15 seconds'.format(str(exc_value)))
            time.sleep(60*15)


def get_thread_recursively(tweet_id):
    list_thread = []
    json = get_single_tweet(tweet_id)  # calling the method for single tweet to get json object

    while True:
        if json.data is None:
            return []
        elif json.data['conversation_id'] == json.data['id']:
            list_thread.append((json.data['id'],
                                json.data['text'],
                                json.includes['users'][0]['username'] + '|' + str(json.data['author_id']),
                                json.data['created_at'].strftime('%Y-%m-%dT%H:%M:%S.000Z')))
            return list_thread

        # getting the next tweet recursively
        else:
            list_thread.append((json.data['id'],
                                json.data['text'],
                                json.includes['users'][0]['username'] + '|' + str(json.data['author_id']),
                                json.data['created_at'].strftime('%Y-%m-%dT%H:%M:%S.000Z')))
            json = get_single_tweet(json.data['referenced_tweets'][0]['id'])


# pattern matching and marking the tweets with the pattern/first/second/3rd person
def recognize_type_of_tweet(thread_pattern, list_thread):
    index_list = []  # output list

    if len(list_thread) == 0:  # if list is empty return
        return index_list

    tweet_text = list_thread[0][1]  # text of cue tweet

    # pattern first person
    first = bool(re.search(r"i(\s*[A-Za-z,;’'\\s@])*\s*being(\s[A-Za-z,;’'\\s@]*)*\ssarcastic", tweet_text.lower())) & \
            ~bool(re.search(r"he(\s*[A-Za-z,;’'\\s@])*\s*being(\s[A-Za-z,;’'\\s@]*)*\ssarcastic", tweet_text.lower())) & \
            ~bool(re.search(r"is*\s*being(\s[A-Za-z,;’'\\s@]*)*\ssarcastic", tweet_text.lower())) & \
            ~bool(re.search(r"you(\s*[A-Za-z,;’'\\s@])*\s*being(\s[A-Za-z,;’'\\s@]*)*\ssarcastic", tweet_text.lower()))
    # pattern second person
    second = bool(re.search(r"you(\s*[A-Za-z,;’'\\s@])*\s*being(\s[A-Za-z,;’'\\s@]*)*\ssarcastic", tweet_text.lower()))
    # pattern third person
    third = (bool(re.search(r"he(\s*[A-Za-z,;'’\\s@])*\s*being(\s[A-Za-z,;’'\\s@]*)*\ssarcastic", tweet_text.lower())) |
             bool(re.search(r"he's being sarcastic", tweet_text.lower())) |
             bool(re.search(r"is*\s*being(\s[A-Za-z,;’'\\s@]*)*\ssarcastic", tweet_text.lower())) |
             bool(re.search(r"they(\s*[A-Za-z,;’'\\s@])*\s*being(\s[A-Za-z,;’'\\s@]*)*\ssarcastic",
                            tweet_text.lower()))) & \
            (~bool(
                re.search(r"you(\s*[A-Za-z,;’'\\s@])*\s*being(\s[A-Za-z,;’'\\s@]*)*\ssarcastic", tweet_text.lower()))) & \
            (~bool(re.search(r"i(\s*[A-Za-z,;’'\\s@])*\s*being(\s[A-Za-z,;’'\\s@]*)*\ssarcastic", tweet_text.lower())))

    # match pattern thread and first person text
    if re.match("(A)([^A]*)(A)([^A]*)$", thread_pattern) and first:
        index_of_A = ([pos for pos, char in enumerate(thread_pattern) if char == 'A'])
        # putting the position of type of tweet in the index list that is returned
        # cue, oblivious, sarcastic, elicit, person class (1, 2 or 3)
        index_list = [0, 1, index_of_A[1], index_of_A[1] + 1, 1]  # last one stands for person (here 1st)

    # match pattern thread and second person text
    elif re.match("(A)A*(B)(A*)$", thread_pattern) and second:
        index_of_B = ([pos for pos, char in enumerate(thread_pattern) if char == 'B'])
        # putting the position of type of tweet in the index list that is returned
        # cue, oblivious, sarcastic, elicit, person class (1, 2 or 3)
        index_list = [0, -1, index_of_B[0], index_of_B[0] + 1, 2]  # last one stands for person (here 2nd)

    # match pattern thread and third person text
    elif re.match("(A)(A*B[AB]*)(C)([AB]*)$", thread_pattern) and third:
        index_of_C = ([pos for pos, char in enumerate(thread_pattern) if char == 'C'])
        # putting the position of type of tweet in the index list that is returned
        # cue, oblivious, sarcastic, elicit, person class (1, 2 or 3)
        index_list = [0, 1, index_of_C[0], index_of_C[0] + 1, 3]  # last one stands for person (here 3rd)

    return index_list


# main function, calling all the others and writing into the dataset
def define_expressions_for_tweet(buffer_txt, input_csv, output_csv):
    with open(buffer_txt, 'a+') as outfile:
        # buffers in case of exceptions
        data_buffer = list()  # buffer for the output data
        df = pd.read_csv(input_csv)  # csv with the cue tweets
        # new dataframe for output
        new_dataset = pd.DataFrame([], columns=["pattern", "person", "cue_id", "sar_id", "obl_id", "eli_id",
                                                "perspective", "cue_text", "sar_text", "obl_text", "eli_text",
                                                "cue_user", "sar_user", "obl_user", "eli_user"])
        # raw data with the json objects
        raw_dataset = pd.DataFrame([], columns=["pattern", "raw_json_replies"])
        index = new_dataset.shape[0]
        raw_index = raw_dataset.shape[0]
        try:
            for i in range(0, df.shape[0]):  # for all tweets in the cue tweet dataset
                tweet_id = df.at[i, 'tweet_id']
                replies_list = get_thread_recursively(tweet_id)  # getting threats for each cue tweet
                data_buffer.append([tweet_id, replies_list])  # write in buffer
                pattern = ''
                author_id_list = {}
                letter = 'A'
                # iterate over replies to get author pattern of thread
                for x in replies_list:
                    current_id = x[2]  # author id of tweet is on pos 2 (0,1,2)
                    if current_id not in author_id_list:
                        author_id_list[
                            current_id] = letter  # tag authors with letters starting with A to get the specific pattern
                        pattern += letter  # concatenate letters (to build whole pattern)
                        letter = next_alpha(letter)  # increase letter (e.g. A -> B)
                    else:
                        pattern += author_id_list[
                            current_id]  # author already occured once in threat just add to pattern

                if len(data_buffer) >= 1000:
                    for data in data_buffer:
                        outfile.write(str(data[0]) + '\t' + str(data[1]) + '\n')
                    data_buffer.clear()

                raw_dataset.loc[raw_index, 'pattern'] = pattern  # add pattern to dataframe
                raw_dataset.loc[
                    raw_index, 'raw_json_replies'] = replies_list  # add json object from request to raw data
                raw_index = raw_index + 1  # increase index (to itertate through all)
                raw_dataset.to_csv('raw_dataset.csv', index=False)  # save raw data

                index_list_types = recognize_type_of_tweet(pattern,
                                                           replies_list)  # now need to add the pattern to dataframe
                print('Index:{}, list:{}, pattern:{}, tweet_list:{}'.format(i, index_list_types, pattern, replies_list))

                if len(index_list_types) < 4 or len(pattern) < 1:  # if there is no person identified scip
                    continue

                new_dataset.loc[index, "pattern"] = pattern

                if index_list_types[
                    4] == 1:  # index 4 bacause here the person information is saved (1,2,3 for 1st, 2nd, 3rd)
                    new_dataset.loc[index, "person"] = '1ST'
                    new_dataset.loc[index, "perspective"] = "INTENDED"
                elif index_list_types[4] == 2:
                    new_dataset.loc[index, "person"] = '2ND'
                    new_dataset.loc[index, "perspective"] = "PERCEIVED"
                elif index_list_types[4] == 3:
                    new_dataset.loc[index, "person"] = '3RD'
                    new_dataset.loc[index, "perspective"] = "PERCEIVED"

                # cue, oblivious, sarcastic, elicit
                cue = replies_list[index_list_types[0]]
                new_dataset.loc[index, "cue_id"] = cue[0]
                new_dataset.loc[index, "cue_text"] = cue[1]
                new_dataset.loc[index, "cue_user"] = cue[2]

                if index_list_types[1] != -1:
                    obl = replies_list[index_list_types[1]]
                    new_dataset.loc[index, "obl_id"] = obl[0]
                    new_dataset.loc[index, "obl_text"] = obl[1]
                    new_dataset.loc[index, "obl_user"] = obl[2]

                sar = replies_list[index_list_types[2]]
                new_dataset.loc[index, "sar_id"] = sar[0]
                new_dataset.loc[index, "sar_text"] = sar[1]
                new_dataset.loc[index, "sar_user"] = sar[2]

                if index_list_types[3] < len(replies_list):
                    eli = replies_list[index_list_types[3]]
                    new_dataset.loc[index, "eli_id"] = eli[0]
                    new_dataset.loc[index, "eli_text"] = eli[1]
                    new_dataset.loc[index, "eli_user"] = eli[2]

                index = index + 1
                new_dataset.to_csv(output_csv, index=False)  # at the end save to new csv


        except IndexError:
            _ = 0

    return


if __name__ == '__main__':
    # parse input
    args = parser.parse_args()
    # method call
    define_expressions_for_tweet(args.buffer_txt, args.input_csv, args.output_csv)