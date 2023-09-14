#!/usr/bin/env python
import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset, Features, Value
from datasets.dataset_dict import DatasetDict
from argparse import ArgumentParser
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, get_scheduler
from sklearn.model_selection import train_test_split
from sarcasm_detection.dataset import AuthorTweetDataset, SarcasticTweetData
from sarcasm_detection.constants import *
from sarcasm_detection.util_functions import *
from sarcasm_detection.util_train import *
from sarcasm_detection.models import SentBertClassifier, SentBertClassifierAttribution
import logging
from util_train import AuthorsEmbedder
import warnings


# first give all random functions SEED
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# ignore future warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

# priming
# python s_bert_with_user.py
# --use_authors=False --author_encoder=priming --sbert_model=sentence-transformers/all-distilroberta-v1 --sbert_dim=768 --user_dim=384 --num_epochs=10 --loss_type=cross_entropy --tweet_sarc_csv=sarc_thread.csv --tweet_non_sarc_csv=non_sarc_thread.csv --bert_tok=sentence-transformers/all-distilroberta-v1 --aut_vocab=user_vocab.pkl --aut_sample=user_sample.pkl --results_dir=transformer --model_name=sbert --class_=all
# user embeddings (average)
#python s_bert_with_user.py
# --use_authors=True --author_encoder=average --sbert_model=sentence-transformers/all-distilroberta-v1 --sbert_dim=768 --user_dim=768 --num_epochs=10 --loss_type=cross_entropy --tweet_sarc_csv=sarc_thread.csv --tweet_non_sarc_csv=non_sarc_thread.csv --bert_tok=sentence-transformers/all-distilroberta-v1 --authors_embedding_path=user_embeddings.pkl --aut_vocab=user_vocab.pkl --results_dir=transformer --model_name=sbert --class_=all
# user embeddings (attribution)
#python s_bert_with_user.py
# --use_authors=True --author_encoder=attribution --sbert_model=sentence-transformers/all-distilroberta-v1 --sbert_dim=768 --user_dim=22169 --num_epochs=10 --loss_type=cross_entropy --tweet_sarc_csv=sarc_thread.csv --tweet_non_sarc_csv=non_sarc_thread.csv --bert_tok=sentence-transformers/all-distilroberta-v1 --authors_embedding_path=user_attribution_prediction.pkl --aut_vocab=user_vocab.pkl --results_dir=transformer --model_name=sbert --class_=all

parser = ArgumentParser()
parser.add_argument("--use_authors", dest="use_authors", type=str2bool)
parser.add_argument("--author_encoder", dest="author_encoder", type=str)
parser.add_argument("--sbert_model", dest="sbert_model", default='sentence-transformers/all-distilroberta-v1', type=str)
parser.add_argument("--authors_embedding_path", dest="authors_embedding_path", type=str)
parser.add_argument("--sbert_dim", dest="sbert_dim", default=768, type=int)
parser.add_argument("--user_dim", dest="user_dim", default=768, type=int) # for sbert average it will be 768 for attribution it corresponds to the number of users in the training set
parser.add_argument("--num_epochs", dest="num_epochs", default=5, type=int)
parser.add_argument("--learning_rate", dest="learning_rate", default=0.0001, type=float)
parser.add_argument("--batch_size", dest="batch_size", default=32, type=int)
parser.add_argument("--loss_type", dest="loss_type", default='cross_entropy', type=str)
parser.add_argument("--tweet_sarc_csv", dest="tweet_sarc_csv", type=str)
parser.add_argument("--tweet_non_sarc_csv", dest="tweet_non_sarc_csv", type=str)
parser.add_argument("--bert_tok", dest="bert_tok", default='sentence-transformers/all-distilroberta-v1', type=str)
parser.add_argument("--aut_vocab", dest="aut_vocab", type=str)
parser.add_argument("--aut_sample", dest="aut_sample", type=str)
parser.add_argument("--results_dir", dest="results_dir", type=str)
parser.add_argument("--model_name", dest="model_name", type=str)
parser.add_argument("--class_", dest="class_", type=str) # all /sarcastic for perceived vs. intended
parser.add_argument("--log", dest="log", type=str) # file/directory name where log file should be saved
#  get current time
TIMESTAMP = get_current_timestamp()


if __name__ == '__main__':
    # parse all user inputs
    args = parser.parse_args()
    USE_AUTHORS = args.use_authors
    author_encoder = args.author_encoder
    s_model = args.sbert_model
    model_name = args.model_name
    authors_embedding_path = args.authors_embedding_path
    S_BERT_DIM = args.sbert_dim
    USER_DIM = args.user_dim
    num_epoch = args.num_epochs
    learning_rate = args.learning_rate
    BATCH_SIZE = args.batch_size
    loss_type = args.loss_type
    sarc_csv = args.tweet_sarc_csv
    non_sarc_csv = args.tweet_non_sarc_csv
    bert_checkpoint = args.bert_tok
    aut_vocab = args.aut_vocab
    aut_sample = args.aut_sample
    results_dir = args.results_dir
    checkpoint_dir = os.path.join(results_dir, f'best_models/{TIMESTAMP}_best_model_sampled.pt')
    class_ = args.class_
    log_ = args.log

    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    handler = logging.FileHandler(f'{log_}_{TIMESTAMP}.log')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            handler,
            logging.StreamHandler()
        ]
    )


    # validate if inputs are correct
    if model_name.lower() != 'sbert':
        logging.info('Model {} not supported. S-Bert is only supported.'.format(model_name))
    if USE_AUTHORS:
        if author_encoder.lower() != 'average' and author_encoder.lower() != 'attribution':
            logging.info('The author encoder {} has been specified incorrect. Can only be "average" or "attribution".'.format(author_encoder))
    else:
        if author_encoder.lower() not in ['none', 'priming']:
            logging.info('The author encoder {} has been specified incorrect. Can only be "none" or "priming".'.format(author_encoder))

    logging.info("Device: {}".format(DEVICE))
    # initialize author vocab (is saved in pickle)
    with open(aut_vocab, 'rb') as vocab:
        AUTHORS_VOCAB = pickle.load(vocab)

    # initialize dataset
    if class_.lower() == 'all':
        dataset = AuthorTweetDataset(AUTHORS_VOCAB, sarc_csv, non_sarc_csv, 2, True) # if cleaned data false else true
    elif class_.lower() == 'sarcastic':
        dataset = SarcasticTweetData(sarc_csv=sarc_csv, punctuations=True, author_vocab=AUTHORS_VOCAB)
    else:
        raise Exception('The class you specified is not supported. Please specify either all or sarcastic.')

    # split data into train, validation, test
    tweet_ids = list(dataset.tweetIdToId.keys())  # get all tweet ids
    labels = list(dataset.tweetIdToLabel.values())  # get all labels to tweet ids
    train_tweets, test_tweets, train_labels, test_labels = train_test_split(tweet_ids, labels, test_size=0.2,
                                                                            random_state=SEED)

    test_tweets, val_tweets, test_labels, val_labels = train_test_split(test_tweets, test_labels, test_size=0.15,
                                                                          random_state=SEED)


    # load author sample if priming should be used
    if author_encoder.lower() == 'priming':
        with open(aut_sample, 'rb') as sample:
            authorToSampledText = pickle.load(sample)
    else:
        authorToSampledText = None


    if USE_AUTHORS and (author_encoder.lower() == 'average' or author_encoder.lower() == 'attribution'):
        embedder = AuthorsEmbedder(embeddings_path=authors_embedding_path, dim=args.user_dim)
    else:
        embedder = None


    # initialize raw data for train, validation, test
    raw_dataset = {'train': {'index': [], 'text': [], 'label': [], 'author_idx': []},
                   'val': {'index': [], 'text': [], 'label': [], 'author_idx': []},
                   'test': {'index': [], 'text': [], 'label': [], 'author_idx': []}}

    # initialize for train
    initialize_raw_data(raw_dataset, dataset, train_tweets, 'train', authorToSampledText, author_encoder, process=True)
    # initialize for validation
    initialize_raw_data(raw_dataset, dataset, val_tweets, 'val', authorToSampledText, author_encoder, process=True)
    # initialize for test
    initialize_raw_data(raw_dataset, dataset, test_tweets, 'test', authorToSampledText, author_encoder, process=True)


    train_size_stats = "Training Size: {}, sarcastic labels {}, non-sarcastic labels {}".format(
        len(raw_dataset['train']['index']),
        raw_dataset['train']['label'].count(1),
        raw_dataset['train']['label'].count(0))
    logging.info(train_size_stats)

    val_size_stats = "Validation Size: {}, sarcastic labels {}, non-sarcastic labels {}".format(
        len(raw_dataset['val']['index']),
        raw_dataset['val']['label'].count(1),
        raw_dataset['val']['label'].count(0))
    logging.info(val_size_stats)

    test_size_stats = "Test Size: {}, sarcastic labels {}, non-sarcastic labels {}".format(
        len(raw_dataset['test']['index']),
        raw_dataset['test']['label'].count(1),
        raw_dataset['test']['label'].count(0))
    logging.info(test_size_stats)


    # initialize model
    # no if else here because we only have s_bert right now
    logging.info("Training with SBERT, model name is {}".format(model_name))
    tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)
    if author_encoder.lower() == 'attribution':
        model = SentBertClassifierAttribution(user_dim=USER_DIM, sbert_model=s_model, sbert_dim=S_BERT_DIM)
    else:
        model = SentBertClassifier(users_layer=USE_AUTHORS, user_dim=USER_DIM, sbert_model=s_model,
                                   sbert_dim=S_BERT_DIM, user_out_dim=BATCH_SIZE)

    # move model to GPU (cuda) if available
    model.to(DEVICE)
    # create a dataset out of the raw data
    ds = DatasetDict()
    for split, d in raw_dataset.items():  # split=('train', 'val', 'test'); d= dictionary containing 'index', 'text',
        # 'label', 'author_idx'
        ds[split] = Dataset.from_dict(mapping=d, features=Features({'label': Value(dtype='int64'),
                                                                    'text': Value(dtype='string'),
                                                                    'index': Value(dtype='int64'),
                                                                    'author_idx': Value(dtype='int64')}))

    logging.info("Tokenizing the dataset.")

    def tokenize_function(example):  # defining function to map to dataset
        return tokenizer(example["text"], truncation=True)


    tokenized_dataset = ds.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    # initialize dataloader to train, validate, test in batches
    train_dataloader = DataLoader(
        tokenized_dataset["train"], batch_size=BATCH_SIZE, collate_fn=data_collator, shuffle=True
    )
    eval_dataloader = DataLoader(
        tokenized_dataset["val"], batch_size=BATCH_SIZE, collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        tokenized_dataset["test"], batch_size=BATCH_SIZE, collate_fn=data_collator
    )


    # initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = num_epoch * len(train_dataloader)
    samples_per_class_train = get_samples_per_class(tokenized_dataset["train"]['labels'])

    # loss function is cross entropy loss
    samples_per_class_train_tensor = torch.tensor(samples_per_class_train).float()
    loss_fn = nn.CrossEntropyLoss(samples_per_class_train_tensor.to(DEVICE))


    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    logging.info("Number of training steps {}".format(num_training_steps))

    progress_bar = tqdm(range(num_training_steps))
    best_accuracy = 0
    best_f1 = 0
    val_metrics = []
    train_loss = []

    # train
    logging.info("Let the training begin...")
    for epoch in range(num_epoch):
        model.train()
        for batch in train_dataloader:
            tweet_index = batch.pop("index")
            author_idx = batch.pop("author_idx")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")


            if USE_AUTHORS and (author_encoder.lower() == 'average' or author_encoder.lower() == 'attribution'):
                authors_embeddings = torch.stack([embedder.embed_author(dataset.idToAuthor[dataset.authorToTweet[index.item()]][0]) for index in tweet_index]).to(DEVICE)
                output = model(batch, authors_embeddings)
            else:
                output = model(batch)


            # loss -> cross_entropy
            loss = loss_fn(output, labels)
            train_loss.append(loss.item())
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)


        val_metric = evaluate_metric(eval_dataloader, model, embedder, USE_AUTHORS, dataset, author_encoder)
        val_metrics.append(val_metric)

        logging.info("Epoch {} **** Metrics validation: {}".format(epoch, val_metric))
        if val_metric['f1_weighted'] > best_f1:
            best_f1 = val_metric['f1_weighted']
            torch.save(model.state_dict(), checkpoint_dir)  # save model

    logging.info('Evaluating')
    model.load_state_dict(torch.load(checkpoint_dir))
    model.to(DEVICE)

    test_metrics = evaluate_metric(test_dataloader, model, embedder, USE_AUTHORS, dataset, author_encoder,
                            True)
    results = test_metrics.pop('results')
    logging.info('Test metrics: {}'.format(test_metrics))

    result_logs = dict()
    result_logs['id'] = TIMESTAMP
    result_logs['seed'] = SEED
    result_logs['sbert_model'] = args.sbert_model
    result_logs['model_name'] = args.model_name
    result_logs['use_authors_embeddings'] = USE_AUTHORS
    result_logs['authors_embedding_path'] = authors_embedding_path
    result_logs['author_encoder'] = author_encoder
    result_logs['train_stats'] = train_size_stats
    result_logs['val_stats'] = val_size_stats
    result_logs['test_stats'] = test_size_stats
    result_logs['epochs'] = num_epoch
    result_logs['optimizer'] = optimizer.defaults
    result_logs["loss_type"] = loss_type
    result_logs['test_metrics'] = test_metrics
    result_logs['checkpoint_dir'] = checkpoint_dir
    result_logs['val_metrics'] = val_metrics
    result_logs['results'] = results

    res_file = os.path.join(results_dir, TIMESTAMP + ".json")
    with open(res_file, mode='w') as f:
        json.dump(result_logs, f, cls=NpEncoder, indent=2)
