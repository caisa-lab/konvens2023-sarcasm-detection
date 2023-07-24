# Sarcasmus detection from Twitter posts

## Requirements:

Create a new conda environment using (replace myenv with the name you want your environment to have):
`conda create -n myenv python=3.8`

When conda asks you to proceed, type y.
This creates the environment in /envs/. No packages will be installed in this environment.

To install required packages, activate the environment:
`conda activate myenv`
And install the packages from the requirements.txt file (if you are using pip3 replace pip with pip3):
`pip install -r requirements.txt`

## Fetching the dataset from Twitter:
Update the CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET and BEARER_TOKEN in the crawling.credentials.py file, if needed. You find a description on how to generate those in the file.

To fetch a new dataset:
* first fetch cue tweets using the crawling.fetch_cue_tweets_client.py
* then filter the cue tweets using the crawling.filter_cue_tweets.py
* then fetch the corresponding sarcastic, elicit and oblivious tweets using the file in which the filtered cue tweets are saved and the crawling.get_sarcastic_thread.py
* then fetch random non-sarcastic tweets using crawling.fetch_non_sarcastic_tweets.py
* to fetch elicit and oblivious tweets corresponding to the non-sarcastic tweets use crawling.get_non_sarcastic_thread.py
* to fetch the user history for the sarcastic and non-sarcastic users use crawling.fetch_user_history.py, once for the saracstic and once for the non-sarcastic users. Afterwards use create_single_vocabulary.py to create a dictionary for sarcastic and non-sarcastic users. To create a dictionary with the combined user history as a dictionary and a sample for the users use create_sum_vocabulary_user_sample.py. If you want to limit the maximum of tweets per user use the corresponding code in create_sum_vocabulary_user_sample.py (inserted as a comment).

## Run text-only models sarcastic vs. non-sarcastic and perceived vs. intended:
You can use 3 different models: bidirectional-LSTM with a glove embedding layer (either max or average pooling), bidirectional-LSTM with a TFIDF embedding (either max or average pooling) or a BERT-sentence transformer.

To use the bidirectional-LSTM with a glove embedding layer use the corresponding glove embedder glove_100d_embedder.pkl or create a new one using the sarcasm_detection.create_glove_embedder.py and download the embedding files from https://nlp.stanford.edu/data/glove.twitter.27B.zip.

To run the text-only models use text_only_baseline_bilstm_sbert.py. If you want to run it for the sarcastic vs. non-sarcastic class, use the csv containing the sarcastic and non-sarcastic tweets and specify class=all. If you want to run it for perceived vs. intended, only use the sarcastic csv and specify class=sarcastic.

## Run models with user context sarcastic vs. non-sarcastic and perceived vs. intended:
Using 3 different models, which take user context in addition to textual information as input.
1. Model using priming: 200 tokens from the user history of each users are added as pre-fix to the tweet text
2. Model using average user embeddings: adding a user embedding based on the average embedding of the historical tweets of each user, using sentence transformers
3. Model using user attribution: adding a user attribution based on the historical tweets using a linear model
4. GNN: TBD

Run sarcasm_detection.s_bert_with_user.py

### Precondition priming:
* filter the sarcastic and non-sarcastic tweets by only keeping tweets for which user history exists (don't delete the original) using create_dataset_with_user_history.py
* create a user vocabulary (create_sum_vocabulary_user_sample.py)


### Precondition average user embeddings:
* filter the sarcastic and non-sarcastic tweets by only keeping tweets for which user history exists (don't delete the original) using create_dataset_with_user_history.py
* Create user embeddings using sarcasm_detection.users_s_bert_embeddings.py

### Precondition user attribution:
#### Create new user embeddings based on user attribution:
1. Create text embeddings for tweets in the sarcastic and non-sarcastic dataset and for the user history using sarcasm_detection.csv_and_user_hist_text_embeddings.py
2. Train the linear model for user attribution using sarcasm_detection.train_user_attribution.py with the text embeddings of the sarcastic and non-sarcastic dataset
3. Extract the user embeddings using sarcasm_detection.user_attribution_extraction.py with the text embeddings of the historical tweets for each user

If you want to run it for the sarcastic vs. non-sarcastic class, use the csv containing the sarcastic and non-sarcastic tweets and specify class=all. If you want to run it for perceived vs. intended, only use the sarcastic csv and specify class=sarcastic.

