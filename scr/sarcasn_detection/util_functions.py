import json
import pickle
import random
import string
from datetime import *
from itertools import zip_longest
import spacy
from sarcasm_detection.constants import SEED, DATETIME_PATTERN
import pandas as pd
from sklearn.utils import shuffle
import argparse
import torch
import re
import emoji
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, balanced_accuracy_score
import numpy as np
import tweepy
from sarcasm_detection.constants import DEVICE

# set seed to random to always get the same token extraction
random.seed(SEED)
np.random.seed(SEED)


# function to create samples of author dictionary containing user id as string and n_token many tokens of user
# history samples
# function takes the following parameters as input:
# dictionary of author_vocab (keys: user ids as strings, value: list of tweets texts)
# n_tokens (int specifying how many tokens should be sampled)
# output_file (string, name of file to save the user dict samples in)
def create_author_samples(author_vocab, n_tokens, output_file_name):
    print("Creating authors sample of " + str(n_tokens) + " tokens.")
    sample_dict = {}
    for key in author_vocab:
        sentence_list = author_vocab[key]  # get len of sentence (word numbers)
        max_number = len(' '.join(sent for sent in sentence_list).split(" "))
        if max_number > n_tokens:
            max_number = n_tokens
        # now sample the words
        random.shuffle(sentence_list)  # to get random selection
        split_random = ' '.join(sent for sent in sentence_list).split(" ")
        random_sent = split_random[0:max_number]
        sample_dict[key] = ' '.join(sent for sent in random_sent)  # add to sample
    with open(output_file_name, 'wb') as f:  # save file to pickle
        pickle.dump(sample_dict, f)
    print("Successfully saved user history sample. Saved for " + str(len(sample_dict)) + " authors")
    print("Author sample is saved with the following file name: " + output_file_name)
    return sample_dict


# function to concat concatenate user history files of sarcastic and non-sarcastic users
def concat_user_hist_files(sarcastic_hist, non_sarcastic_hist):
    with open(sarcastic_hist) as file_1:
        sarcastic_lines = file_1.readlines()
    with open(non_sarcastic_hist) as file_2:
        non_sarcastic_lines = file_2.readlines()
    user_hist_list = sarcastic_lines + non_sarcastic_lines
    return user_hist_list


# function to merge sarcastic and not sarcastic data
# gets file name of csv files as input
def merge_datasets(sarc_data, non_sarc_data):
    dataset_sarc = pd.read_csv(sarc_data)
    dataset_non_sarc = pd.read_csv(non_sarc_data)

    perspective = list(map(lambda x: x.lower(), dataset_sarc['perspective'].tolist()))
    dataset_sarc = dataset_sarc.drop(['cue_text', 'cue_id', 'pattern', 'person', 'cue_user', 'perspective'],
                                     axis=1)

    # add column with perspective to none sarcastic and sarcastic
    dataset_non_sarc['perspective'] = "not_sarcastic"
    dataset_sarc['perspective'] = perspective

    # merge dataframe ignoring index
    frames = [dataset_non_sarc, dataset_sarc]
    new_frame = pd.concat(frames, ignore_index=True)
    # shuffle the merged dataframe
    new_frame = shuffle(new_frame, random_state=SEED)

    # save file as csv
    # new_frame.to_csv('SPIRS-merged.csv', index=False)
    return new_frame


# function to print information about user input
def print_args(args, logger):
    for arg in vars(args):
        print("{} \t \t {}".format(args, getattr(args, arg)))


# Count the frequency of each value in labels
def get_samples_per_class(labels):
    return torch.bincount(labels).tolist()


# function to get current time
def get_current_timestamp():
    return str(datetime.now()).replace(" ", "_").replace(".", ":")


# function to transform time stamp to sting
def timestamp_to_string(timestamp):
    dt = datetime.utcfromtimestamp(timestamp)
    return dt.strftime(DATETIME_PATTERN)


def get_and_print_metrics(gold, predictions):
    cm = confusion_matrix(gold, predictions)
    print(cm)
    f1Score_1 = f1_score(gold, predictions, average='macro')
    print("Total f1 score macro {:3f}: ".format(f1Score_1))
    f1Score_2 = f1_score(gold, predictions, average='micro')
    print("Total f1 score micro {:3f}:".format(f1Score_2))
    f1Score_4 = f1_score(gold, predictions, average='weighted')
    print("Total f1 score weighted {:3f}:".format(f1Score_4))
    accuracy = accuracy_score(gold, predictions)
    print("Accuracy {:3f}:".format(accuracy))
    bal_accuracy = balanced_accuracy_score(gold, predictions)
    print("Balanced accuracy {:3f}:".format(bal_accuracy))

    return {'macro': f1Score_1, 'micro': f1Score_2, 'weighted': f1Score_4, 'accuracy': accuracy,
            'balanced accuracy': bal_accuracy, 'cm': cm}


def get_metrics(gold, predictions):
    return {'macro': f1_score(gold, predictions, average='macro'),
            'micro': f1_score(gold, predictions, average='micro'),
            'weighted': f1_score(gold, predictions, average='weighted'),
            'balanced accuracy': balanced_accuracy_score(gold, predictions),
            'accuracy': accuracy_score(gold, predictions), 'cm': confusion_matrix(gold, predictions)}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


EMOJI_DESCRIPTION_SCRUB = re.compile(r':(\S+?):')
HASHTAG_BEFORE = re.compile(r'#(\S+)')
BAD_HASHTAG_LOGIC = re.compile(r'(\S+)!!')
FIND_MENTIONS = re.compile(r'@(\S+)')
LEADING_NAMES = re.compile(r'^\s*((?:@\S+\s*)+)')
TAIL_NAMES = re.compile(r'\s*((?:@\S+\s*)+)$')


# function to clean text
def process_tweet(s, save_text_formatting=True, keep_emoji=False, keep_usernames=False):
    if save_text_formatting:
        # remove links
        s = re.sub(r'https\S+', r'', str(s))
        s = re.sub(r'http\S+', r'', str(s))
    else:
        s = re.sub(r'http\S+', r'', str(s))
        s = re.sub(r'https\S+', r'', str(s))
        s = re.sub(r'x{3,5}', r'', str(s))
    # remove unwanted strings
    s = re.sub(r'\\n', ' ', s)
    s = re.sub(r'\s', ' ', s)
    s = re.sub(r'<br>', ' ', s)
    s = re.sub(r'&amp;', '&', s)
    s = re.sub(r'&#039;', "'", s)
    s = re.sub(r'&gt;', '>', s)
    s = re.sub(r'&lt;', '<', s)
    s = re.sub(r'\'', "'", s)

    if save_text_formatting:
        s = emoji.demojize(s)
    elif keep_emoji:
        s = emoji.demojize(s)
        s = s.replace('face_with', '')
        s = s.replace('face_', '')
        s = s.replace('_face', '')
        s = re.sub(EMOJI_DESCRIPTION_SCRUB, r' \1 ', s)
        s = s.replace('(_', '(')
        s = s.replace('_', ' ')

    s = re.sub(r"\\x[0-9a-z]{2,3,4}", "", s)

    if save_text_formatting:
        s = re.sub(HASHTAG_BEFORE, r'\1!!', s)
    else:
        s = re.sub(HASHTAG_BEFORE, r'\1', s)
        s = re.sub(BAD_HASHTAG_LOGIC, r'\1', s)

    if not save_text_formatting:
        if keep_usernames:
            s = ' '.join(s.split())

            s = re.sub(LEADING_NAMES, r' ', s)
            s = re.sub(TAIL_NAMES, r' ', s)

            s = re.sub(FIND_MENTIONS, r'\1', s)
        else:
            s = re.sub(FIND_MENTIONS, r' ', s)
    user_regex = r".?@.+?( |$)|<@mention>"
    s = re.sub(user_regex, " @user ", s, flags=re.I)

    # Just in case -- remove any non-ASCII and unprintable characters, apart from whitespace
    s = "".join(x for x in s if (x.isspace() or (31 < ord(x) < 127)))
    s = ' '.join(s.split())
    return s


# function to initialize raw data
# raw_data: dict with train, validation, test. Each with index, text, label, author_idx
# dataset: a AuthorTweetDataset
# tweet_id_list: list of tweet ids
# modus: 'train', 'val', 'test'
# author_sample: dict with author samples
def initialize_raw_data(raw_data, dataset, tweet_id_list, modus, author_sample, author_encoder, process=True,
                        save_text_formatting=True, keep_emoji=False, keep_usernames=False, text_len=False,
                        tokenizer=None):
    nlp = spacy.load('en_core_web_sm')
    debug = True
    if text_len:
        raw_data[modus]['text_len'] = []
    for tweet_id in tweet_id_list:
        author_id = dataset.authorToTweet[tweet_id]
        author_id_string = dataset.authorIdToId[author_id]
        label = dataset.tweetIdToLabel[tweet_id]

        if process:
            text = process_tweet(dataset.idToCleanText[tweet_id], save_text_formatting, keep_emoji, keep_usernames)
        else:
            text = dataset.idToCleanText[tweet_id]

        # check if user exists in author_vocabulary -> if not the entry is discarded
        if dataset.author_vocab is not None and author_id_string in dataset.author_vocab:
            raw_data[modus]['index'].append(tweet_id)
            raw_data[modus]['author_idx'].append(author_id)
            raw_data[modus]['label'].append(label)
            if author_encoder.lower() == 'priming':
                priming_text = author_sample[author_id_string]
                # print(priming_text + ' [SEP] ' + text)
                raw_data[modus]['text'].append(priming_text + ' [SEP] ' + text)
            elif author_sample is None and author_encoder.lower() == 'average':
                raw_data[modus]['text'].append(text)
            else:
                raw_data[modus]['text'].append(text)
        else:
            raw_data[modus]['index'].append(tweet_id)
            raw_data[modus]['author_idx'].append(author_id)
            raw_data[modus]['label'].append(label)

            if text_len:
                len_ = len(text.split(" "))
                raw_data[modus]['text_len'].append(len_)

            raw_data[modus]['text'].append(text)


def initialize_data_eliciting(raw_data, dataset, tweet_id_list, modus, process=True,
                              save_text_formatting=True, keep_emoji=False, keep_usernames=False):
    for tweet_id in tweet_id_list:
        author_id = dataset.authorToTweet[tweet_id]
        author_id_string = dataset.authorIdToId[author_id]
        label = dataset.tweetIdToLabel[tweet_id]
        df = dataset.dataframe
        eliciting = list(df[df['sar_id'] == dataset.idToTweetId[tweet_id]]['eli_text'])[0]
        if not isinstance(eliciting, float):
            eliciting = process_tweet(eliciting, save_text_formatting, keep_emoji, keep_usernames)
            text = process_tweet(dataset.idToCleanText[tweet_id], save_text_formatting, keep_emoji, keep_usernames)

            raw_data[modus]['index'].append(tweet_id)
            raw_data[modus]['author_idx'].append(author_id)
            raw_data[modus]['label'].append(label)
            raw_data[modus]['text'].append(eliciting + ' [SEP] ' + text)

def initialize_data_oblivious(raw_data, dataset, tweet_id_list, modus, process=True,
                              save_text_formatting=True, keep_emoji=False, keep_usernames=False):
    for tweet_id in tweet_id_list:
        author_id = dataset.authorToTweet[tweet_id]
        author_id_string = dataset.authorIdToId[author_id]
        label = dataset.tweetIdToLabel[tweet_id]
        df = dataset.dataframe
        oblivious = list(df[df['sar_id'] == dataset.idToTweetId[tweet_id]]['obl_text'])[0]
        if not isinstance(oblivious, float):
            oblivious = process_tweet(oblivious, save_text_formatting, keep_emoji, keep_usernames)
            text = process_tweet(dataset.idToCleanText[tweet_id], save_text_formatting, keep_emoji, keep_usernames)

            raw_data[modus]['index'].append(tweet_id)
            raw_data[modus]['author_idx'].append(author_id)
            raw_data[modus]['label'].append(label)
            raw_data[modus]['text'].append(oblivious + ' [SEP] ' + text)
def initialize_data_eliciting_and_oblivious(raw_data, dataset, tweet_id_list, modus, process=True,
                              save_text_formatting=True, keep_emoji=False, keep_usernames=False):
    for tweet_id in tweet_id_list:
        author_id = dataset.authorToTweet[tweet_id]
        author_id_string = dataset.authorIdToId[author_id]
        label = dataset.tweetIdToLabel[tweet_id]
        df = dataset.dataframe
        eliciting = list(df[df['sar_id'] == dataset.idToTweetId[tweet_id]]['eli_text'])[0]
        oblivious = list(df[df['sar_id'] == dataset.idToTweetId[tweet_id]]['obl_text'])[0]
        if not isinstance(eliciting, float) and not isinstance(oblivious, float):
            eliciting = process_tweet(eliciting, save_text_formatting, keep_emoji, keep_usernames)
            oblivious = process_tweet(oblivious, save_text_formatting, keep_emoji, keep_usernames)
            text = process_tweet(dataset.idToCleanText[tweet_id], save_text_formatting, keep_emoji, keep_usernames)

            raw_data[modus]['index'].append(tweet_id)
            raw_data[modus]['author_idx'].append(author_id)
            raw_data[modus]['label'].append(label)
            raw_data[modus]['text'].append(oblivious + ' ' + eliciting + ' [SEP] ' + text)

def initialize_raw_data_tweet(raw_data, dataset, tweet_id_list, modus, process=True, save_text_formatting=True,
                              keep_emoji=False, keep_usernames=False, text_len=False, tokenizer=None):
    if text_len:
        raw_data[modus]['text_len'] = []
    for tweet_id in tweet_id_list:
        author_id = dataset.authorToTweet[tweet_id]
        author_id_string = dataset.authorIdToId[author_id]
        label = dataset.tweetIdToLabel[tweet_id]
        if process:
            text = process_tweet(dataset.idToCleanText[tweet_id], save_text_formatting, keep_emoji, keep_usernames)
        else:
            text = dataset.idToCleanText[tweet_id]

        raw_data[modus]['index'].append(tweet_id)
        raw_data[modus]['author_idx'].append(author_id)
        raw_data[modus]['label'].append(label)
        if process:
            text = process_tweet(dataset.idToCleanText[tweet_id], save_text_formatting, keep_emoji, keep_usernames)
        else:
            text = dataset.idToCleanText[tweet_id]
        if text_len:
            len_ = len(text.split(" "))
            raw_data[modus]['text_len'].append(len_)

        raw_data[modus]['text'].append(text)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# function to authenticate the Twitter api consumer_key = generated in the Twitter dev section of your account
# consumer_secret = generated in the Twitter dev section of your account access_token = generated in the Twitter dev
# section of your account access_secret = generated in the Twitter dev section of your account wait_on_rate_limit =
# True/False. If True api functions will be paused when rate limit has been exceeded (error 429 occurs)
def authenticate_api(consumer_key, consumer_secret, access_token, access_secret, wait_on_rate_limit):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    return tweepy.API(auth, wait_on_rate_limit=wait_on_rate_limit)

def extract_batches(seq, batch_size=32):
    n = len(seq) // batch_size
    batches = []
    for i in range(n):
        batches.append(seq[i * batch_size:(i + 1) * batch_size])
    if len(seq) % batch_size != 0:
        batches.append(seq[n * batch_size:])
    return batches


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return list(zip_longest(*args, fillvalue=fillvalue))
