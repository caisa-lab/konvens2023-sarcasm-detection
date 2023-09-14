from tqdm import tqdm
import string
from sarcasm_detection.util_functions import merge_datasets
import pandas as pd

def initial_clean_text(text, punctuation):
    clean_text = str(text).lower()  # lower
    if punctuation:
        clean_text = clean_text.translate(str.maketrans('', '', string.punctuation.replace('@', '')))  # remove punctuations
    clean_text = clean_text.replace('\n', ' ') # removing linebreaks
    clean_text = clean_text.replace('  ', ' ')
    return clean_text

class AuthorTweetDataset:
    """ Creates the datastructure for a data set containing author and tweet information """
    def __init__(self, author_vocab, sarc_csv, non_sarc_csv, dim, punctuations):
        self.punctuation = punctuations
        self.author_vocab = author_vocab  # author vocabulary (key: author id, value: user hist)
        self.authorIdToId = dict()  # key: number (int), value: author id  (str)
        self.idToAuthor = list()  # lists of author id (string) and author name (string), index of id == number from
        # authorIdToId

        self.tweetIdToId = dict()  # number (int) as key, tweet id (string) as value
        self.idToTweetId = list()  # tweet id (string), index of id == int from tweetIdToId
        self.idToText = dict()  # key: tweet id (int), value: tweet text (string)
        self.idToCleanText = dict()  # key: tweet id (int), value: tweet text without punctuations and lowered (string)

        self.tweetIdToLabel = dict()  # key: tweet id (int), value: encoded label to tweet (int)
        self.authorToLabel = dict()  # key: author id (int), value: encoded label to tweet (int)

        self.tweetToAuthor = dict()  # key: (int) author id, value: tweet id (int)
        self.authorToTweet = dict()  # key: (int) tweet id, value: author id (int)
        self.stringIdToCleanText = dict() # key: (str) twee id, value: clean text (str)
        self.uniqueAuthorsVocab = dict()
        self.tweetToAuthorString = dict() # tweet id string: authors id string

        if dim == 3:
            self.labelToEncoding = {"not_sarcastic": 0, "intended": 1, "perceived": 2}
            self.encodingToLabel = {0: "not_sarcastic", 1: "intended", 2: "perceived"}

        elif dim == 2:
            self.labelToEncoding = {"not_sarcastic": 0, "intended": 1, "perceived": 1}
            self.encodingToLabel = {0: "not_sarcastic", 1: "sarcastic"}

        self.dataframe = merge_datasets(sarc_csv, non_sarc_csv)
        self.load_data()

    def load_data(self):
        """ Iterates through dataframe and fills dictionaries and lists"""
        self.dataframe = self.dataframe.reset_index()  # to make sure that we start with 0 and enumerate correctly
        for i, row in tqdm(self.dataframe.iterrows(), desc="Creating dictionary maps for the dataset"):
            split_user = str(row["sar_user"]).split("|")  # some user only contain either name or author id not both
            if len(split_user) > 1:  # users containing both
                self.authorIdToId[i] = split_user[1]
                self.idToAuthor.append([split_user[1], split_user[0]])
                self.tweetToAuthorString[row['sar_id']] = split_user[1]
            else:
                self.authorIdToId[i] = split_user[0]
                self.idToAuthor.append([split_user[0]])
                self.tweetToAuthorString[row['sar_id']] = split_user[0]

            self.tweetIdToId[i] = row["sar_id"]  # enumerates tweet id
            self.idToTweetId.append(row["sar_id"])
            self.idToText[i] = row["sar_text"]
            self.tweetIdToLabel[i] = self.labelToEncoding[row["perspective"]]
            self.authorToLabel[i] = self.labelToEncoding[row["perspective"]]
            self.tweetToAuthor[i] = i
            self.authorToTweet[i] = i

        for ids, text in tqdm(self.idToText.items(), desc="Lowering twitter text and removing punctuations"):
            clean_text = initial_clean_text(text, punctuation=self.punctuation)
            self.idToCleanText[ids] = clean_text
            self.stringIdToCleanText[self.tweetIdToId[ids]] = clean_text
            # adding author to tweets mapping
            author = str(list(self.dataframe.loc[self.dataframe['sar_id'] == self.tweetIdToId[ids]]['sar_user'])[0]).split('|')
            if len(author) > 1:
                author_id = author[1]
            else:
                author_id = author[0]
            if author_id in self.uniqueAuthorsVocab:
                self.uniqueAuthorsVocab[author_id].append([self.tweetIdToId[ids],clean_text])
            else:
                self.uniqueAuthorsVocab[author_id] = [[self.tweetIdToId[ids],clean_text]]


class SarcasticTweetData:
    """
    Class for sarcastic data, containing intended versus perceived sarcastic tweets.
    """
    def __init__(self, sarc_csv, punctuations, author_vocab):
        self.author_vocab = author_vocab
        self.punctuation = punctuations
        self.authorIdToId = dict()  # key: number (int), value: author id  (str)
        self.idToAuthor = list()  # lists of author id (string) and author name (string), index of id == number from
        # authorIdToId

        self.tweetIdToId = dict()  # number (int) as key, tweet id (string) as value
        self.idToTweetId = list()  # tweet id (string), index of id == int from tweetIdToId
        self.idToText = dict()  # key: tweet id (int), value: tweet text (string)
        self.idToCleanText = dict()  # key: tweet id (int), value: tweet text without punctuations and lowered (string)

        self.tweetIdToLabel = dict()  # key: tweet id (int), value: encoded label to tweet (int)
        self.authorToLabel = dict()  # key: author id (int), value: encoded label to tweet (int)

        self.tweetToAuthor = dict()  # key: (int) author id, value: tweet id (int)
        self.authorToTweet = dict()  # key: (int) tweet id, value: author id (int)

        self.labelToEncoding = {"INTENDED": 0, "PERCEIVED": 1}
        self.encodingToLabel = {0: "INTENDED", 1: "PERCEIVED"}

        self.dataframe = pd.read_csv(sarc_csv)
        self.stringIdToCleanText = dict()
        self.uniqueAuthorsVocab = dict()
        self.tweetToAuthorString = dict()  # tweet id string: authors id string
        self.load_data()

    def load_data(self):
        """ Iterates through dataframe and fills dictionaries and lists"""
        self.dataframe = self.dataframe.reset_index()  # to make sure that we start with 0 and enumerate correctly
        for i, row in tqdm(self.dataframe.iterrows(), desc="Creating dictionary maps for the dataset"):
            split_user = str(row["sar_user"]).split("|")  # some user only contain either name or author id not both
            if len(split_user) > 1:  # users containing both
                self.authorIdToId[i] = split_user[1]
                self.idToAuthor.append([split_user[1], split_user[0]])
                self.tweetToAuthorString[row['sar_id']] = split_user[1]
            else:
                self.authorIdToId[i] = split_user[0]
                self.idToAuthor.append([split_user[0]])
                self.tweetToAuthorString[row['sar_id']] = split_user[0]

            self.tweetIdToId[i] = row["sar_id"]  # enumerates tweet id
            self.idToTweetId.append(row["sar_id"])
            self.idToText[i] = row["sar_text"]
            self.tweetIdToLabel[i] = self.labelToEncoding[row["perspective"]]
            self.authorToLabel[i] = self.labelToEncoding[row["perspective"]]
            self.tweetToAuthor[i] = i
            self.authorToTweet[i] = i

        for ids, text in tqdm(self.idToText.items(), desc="Lowering twitter text and removing punctuations"):
            clean_text = initial_clean_text(text, punctuation=self.punctuation)
            self.idToCleanText[ids] = clean_text
            self.stringIdToCleanText[self.tweetIdToId[ids]] = clean_text
            # adding author to tweets mapping
            author = str(list(self.dataframe.loc[self.dataframe['sar_id'] == self.tweetIdToId[ids]]['sar_user'])[0]).split('|')
            if len(author) > 1:
                author_id = author[1]
            else:
                author_id = author[0]
            if author_id in self.uniqueAuthorsVocab:
                self.uniqueAuthorsVocab[author_id].append([self.tweetIdToId[ids], clean_text])
            else:
                self.uniqueAuthorsVocab[author_id] = [[self.tweetIdToId[ids], clean_text]]

