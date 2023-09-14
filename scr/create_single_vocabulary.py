import string

import pandas as pd
from argparse import ArgumentParser
import re
import pickle

parser = ArgumentParser()
parser.add_argument("--csv_1", dest="csv_1", type=str)
parser.add_argument("--csv_2", dest="csv_2", type=str)
parser.add_argument("--user_file_1", dest="user_file_1", type=str)
parser.add_argument("--user_file_2", dest="user_file_2", type=str)
parser.add_argument("--user_out_1", dest="user_out_1", type=str)
parser.add_argument("--user_out_2", dest="user_out_2", type=str)
# python create_single_vocabulary.py --csv_1=sarc_thread.csv --csv_2=non_sarc_thread.csv --user_file_1=user_hist_new_sarc.txt user_file_2=user_hist_new_non_sarc.txt
# --user_out_1=user_hist_single_sarc.pkl --user_out_2=user_hist_single_non_sarc.pkl

def create_separate_author_vocab(csv, user_hist, output_file_name):
    df = pd.read_csv(csv)
    user_id_list = []
    author_vocab = {}

    # first create a list of user ids
    for i, row in df.iterrows():
        user_tag = row['sar_user'].split("|")[1]
        user_id_list.append(user_tag)
    print('created list of user ids')

    # create author vocabulary dictionary
    with open(user_hist, 'r') as hist:
        for lines in hist:
            line = lines.split('\t')
            user_id = line[0]
            user_text = str(line[2]).lower()
            user_text = re.sub(r'\s+', ' ', user_text.replace('\n', ' '))
            user_text = user_text.translate(str.maketrans('', '', string.punctuation))

            # initialize author vocab
            if user_id in user_id_list:
                print(user_id, " ", user_text)
                if user_id in author_vocab:
                    if user_text:
                        author_vocab[user_id].append(user_text)
                else:
                    if user_text:
                        new_text_list = [user_text]
                        author_vocab[user_id] = new_text_list

    print('Finished creating author vocabulary')
    with open(output_file_name, 'wb') as f:  # save file to pickle
        pickle.dump(author_vocab, f)
    return author_vocab

if __name__ == '__main__':
    args = parser.parse_args()
    author_vocab_1 = create_separate_author_vocab(args.csv_1, args.user_file_1, args.user_out_1)
    author_vocab_2 = create_separate_author_vocab(args.csv_2, args.user_file_2, args.user_out_2)

    print("Created two author vocabularies. The first has a length of: {} and the second: {}".format(len(author_vocab_1), len(author_vocab_2)))







