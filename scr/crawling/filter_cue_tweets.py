#!/usr/bin/env python

import argparse
import pandas as pd
import warnings

# let script run like: python filter_cue_tweets.py "fetched_cue_tweets_client.csv" "filtered_cue_tweets_client.csv"
parser = argparse.ArgumentParser(description='Filters cue tweets')
parser.add_argument('unfiltered_file', help='CSV file containing unfiltered cue tweets', type=str)
parser.add_argument('filtered_file', help='CSV file containing filtered cue tweets', type=str)

# ignore warning
warnings.filterwarnings("ignore")


# method to filter cue tweets
def filter_data(dataframe_unfiltered):
    """
    Filters not sarcastic cue tweets and tweets towards recipients were undecided if authors of a tweet were
    sarcastic. Filtering is performed using patterns. Patterns have been defined based on a previous manual analysis
    of 200 cue tweets.
    :param dataframe_unfiltered: name of csv file containing unfiltered cue tweets, crawled beforehand (str).
    :return: returns a filtered pandas dataframe
    """
    # temp df to lower the case and filter than
    tem_df = dataframe_unfiltered
    text_lower = tem_df['text'].str.lower().tolist()
    tem_df['text'] = text_lower
    df_exclude = tem_df[(tem_df.text.str.contains(r"not being sarcastic")) |
                        (tem_df.text.str.contains(r"(sarcastic)\s*(\?)+")) |
                        (tem_df.text.str.contains(r"sarcastic\sor")) |
                        (tem_df.text.str.contains(r"pray(\s*[A-Za-z,;'\\s@])*\s*being sarcastic")) |
                        (tem_df.text.str.contains(r"hope(\s*[A-Za-z,;'\\s@])*\s*being sarcastic")) |
                        (tem_df.text.str.contains(r"if(\s*[A-Za-z,;'\\s@])*\s*being sarcastic")) |
                        (tem_df.text.str.contains(r"sarcastic[A-Za-z,;’\\s@]*\s*correct")) |
                        (tem_df.text.str.contains(
                            r"hope(\s*[A-Za-z,;'\\s@])*\s*being(\s*[A-Za-z,;'\\s@])*\s*sarcastic")) |
                        (tem_df.text.str.contains(r"sarcastic\s*([A-Za-z,;’\\s@]\s){0,2}right")) |
                        (tem_df.text.str.contains(r"not(\s*[A-Za-z,;'\\\/s@])*\s*sarcastic")) |
                        (tem_df.text.str.contains(r"wasn't being sarcastic")) |
                        (tem_df.text.str.contains(r"wasnt being sarcastic")) |
                        (tem_df.text.str.contains(r"wasn’t being sarcastic")) |
                        (tem_df.text.str.contains(r'hope you’re being sarcastic')) |
                        (tem_df.text.str.contains(r'are you being sarcastic')) |
                        (tem_df.text.str.contains(r"was not being sarcastic")) |
                        (tem_df.text.str.contains(r"weren't being sarcastic")) |
                        (tem_df.text.str.contains(r'weren’t being sarcastic')) |
                        (tem_df.text.str.contains(r"werent being sarcastic")) |
                        (tem_df.text.str.contains(r"were not being sarcastic"))
                        ]

    cond = dataframe_unfiltered['tweet_id'].isin(df_exclude['tweet_id'])
    dataframe_unfiltered.drop(dataframe_unfiltered[cond].index, inplace=True)
    print("Data has been successfully filtered down to: " + str(len(dataframe_unfiltered)))
    return dataframe_unfiltered

if __name__ == '__main__':
    # parse input
    args = parser.parse_args()
    # create dataframe out of CSV
    dataframe_unfiltered = pd.read_csv(args.unfiltered_file)
    # method call
    data_filtered = filter_data(dataframe_unfiltered)
    # save as CSV
    data_filtered.to_csv(args.filtered_file, index=False)
