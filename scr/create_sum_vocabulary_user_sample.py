import string
from argparse import ArgumentParser
import pickle
import random
from sarcasm_detection.constants import SEED
from sarcasm_detection.util_functions import create_author_samples
import re

random.seed(SEED)
parser = ArgumentParser()

# execute the python file:
# python create_sum_vocabulary_user_sample.py
# --hist_sarc=user_hist_single_sarc.pkl --hist_non_sarc=user_hist_single_non_sarc.pkl
# --output_vocab=user_vocab.pkl --output_sample=user_sample.pkl --token=100

parser.add_argument("--hist_sarc", dest="hist_sarc",
                    help="Name of file containing the sarcastic author dictionary",
                    type=str)
parser.add_argument("--hist_non_sarc", dest="hist_non_sarc",
                    help="Name of file containing the non-sarcastic author dictionary",
                    type=str)
parser.add_argument("--output_vocab", dest="output_vocab",
                    help="File name for new pkl file containing the author vocabulary",
                    type=str)
parser.add_argument("--output_sample", dest="output_sample",
                    help="File name for new pkl file containing the author sample",
                    type=str)
parser.add_argument("--token", dest="token",
                    help="Number of token to sample",
                    type=int)

def create_sum_dictionary(dict_sarc, dict_non_sarc):
    """
    creates a dictionary out of two existing dictionaries by merging them but also merging their values if both dictionaries have the same keys.
    :param dict_sarc: name of pickle file containing the dictionary of sarcastic users (str)
    :param dict_non_sarc: name of pickle file containing the dictionary of non-sarcastic users (str)
    :return: a merged dictionary, the original sarcastic dictionary, the original non-sarcastic dictionary
    """
    # load the dictionaries
    with open(dict_sarc, 'rb') as sarc_dict:
        sar_dict = pickle.load(sarc_dict)
    with open(dict_non_sarc, 'rb') as non_sarc_dict:
        non_sar_dict = pickle.load(non_sarc_dict)


    print('len of {}: {}; len of {}: {}'.format(len(sar_dict), dict_sarc, len(non_sar_dict), dict_non_sarc))
    # check if they have intersecting keys because we need to create a union of the history before updating the dictionaries to one
    intersection = set(sar_dict.keys()) & set(non_sar_dict.keys()) ## check if they have intersecting keys
    if intersection: # intersection is not empty -> there are users which are in both dictionaries
        print("{} of the authors are in both dictionaries. They have the following user ids:\n{}".format(len(intersection), intersection))
        for users in intersection: # create a union of the histories, only keeping unique tweets
            unique_union = set(sar_dict[users] + non_sar_dict[users])
            non_sar_dict[users] = list(unique_union) # overwrite the values because we will use non_sarc_dict for the update
        sar_dict.update(non_sar_dict)
    else:
        print('No intersection of authors between both dictionaries.')
        sar_dict.update(non_sar_dict)

    return sar_dict

# In case the user dictionary contains to many tweets per user
def sample_max_number_of_tweets(user_dict, max_number):
    user_dict_ = user_dict.copy()
    for user in user_dict_:
        tweet_list = user_dict_[user]
        if len(tweet_list) > max_number:
            tweet_list = random.sample(tweet_list, max_number)
            user_dict_[user] = tweet_list
    return user_dict_


if __name__ == '__main__':
    # parse all user inputs
    args = parser.parse_args()
    hist_sarc = args.hist_sarc
    hist_non_sarc = args.hist_non_sarc
    output_vocab = args.output_vocab
    output_sample = args.output_sample
    token = args.token

    sum_dict =  create_sum_dictionary(hist_sarc, hist_non_sarc)

    # save the dictionary for all users (sarcastic and non-sarcastic)
    print('Saving the merged dictionary...')
    with open(output_vocab, 'wb') as handle:
        pickle.dump(sum_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    """
    #  In case the user dictionary contains to many tweets per user
    print('Load user dictionary')
    with open(output_vocab,'rb') as _dict:
        sum_dict = pickle.load(_dict)

    print('Create user vocab with 500 tweets per user as upper bound.')
    sum_dict_500 = sample_max_number_of_tweets(sum_dict, 500)
    sum_dict_500_name = 'user_vocab_500.pkl' # change the name of the file, if needed

    print("Saving user dictionaries to pkl")
    with open(sum_dict_500_name, 'wb') as f:  # save file to pickle
        pickle.dump(sum_dict_500, f)


    print('Creating user samples')
    user_sample_500_name = 'user_sample_500.pkl' # change the name of the file, if needed
    out1 = create_author_samples(sum_dict_500, 200,  user_sample_500_name)
    """


    print('Creating and saving sample..')
    out = create_author_samples(sum_dict, token,  output_sample)

    print('Successfully saved all data')









