# Personalized Intended and Perceived Sarcasm Detection on Twitter

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
You can use the BERT-sentence transformer either only with textual features (using text_only_baseline_sbert.py) or with addtional conversational features (adding eliciting and oblivious tweets - using text_only_eliciting_oblivous.py)

If you want to run the models for the sarcastic vs. non-sarcastic class, use the csv containing the sarcastic and non-sarcastic tweets and specify class=all. If you want to run it for perceived vs. intended, only use the csv containing sarcastic tweets only and specify class=sarcastic.

## Run models with user context sarcastic vs. non-sarcastic and perceived vs. intended:
Using 3 different models, which take user context in addition to textual information as input.
1. Model using priming: 200 tokens from the user history of each users are added as pre-fix to the tweet text
2. Model using average user embeddings: adding a user embedding based on the average embedding of the historical tweets of each user, using sentence transformers
3. Model using user attribution: adding a user attribution based on the historical tweets using a linear model
4. GNN: modeling the social relations between users, and the relations between tweets and users. For this purpose,a heterogeneous graph G = (V, E) is build, where V = {U ∪ T}, which consists of two types of nodes: users and tweets.

To use the models described in 1. - 3. run sarcasm_detection.s_bert_with_user.py and specify the run configuration accordingly (see examplary calls in the file).
To use the model described in 4.: TBD

Models described in 2. & 3. can be additionally enhanced with conversational features as described in "Run text-only models", by appending the eliditing and oblivous tweets to the sarcastic tweet.

### Precondition priming:
* create a user vocabulary using create_sum_vocabulary_user_sample.py

### Precondition average user embeddings:
* Create user embeddings using the historical tweets of each author in the sarcastic and non-sarcastic dataset with sarcasm_detection.users_s_bert_embeddings.py

### Precondition user attribution:
#### Create new user embeddings based on user attribution:
1. Create text embeddings for tweets in the sarcastic and non-sarcastic dataset and for the user history using sarcasm_detection.csv_and_user_hist_text_embeddings.py
2. Train the linear model for user attribution using sarcasm_detection.train_user_attribution.py with the text embeddings of the sarcastic and non-sarcastic dataset (only using the training set)
3. Extract the user embeddings using sarcasm_detection.user_attribution_extraction.py with the text embeddings of the historical tweets for each user

If you want to run it for the sarcastic vs. non-sarcastic class, use the csv containing the sarcastic and non-sarcastic tweets and specify class=all. If you want to run it for perceived vs. intended, only use the sarcastic csv and specify class=sarcastic.


