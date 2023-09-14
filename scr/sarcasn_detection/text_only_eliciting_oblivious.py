import os
import pickle
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
from sarcasm_detection.models import SentBertClassifier
import logging
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer

# first give all random functions SEED
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# call python text_only_eliciting_oblivious.py
# --num_epochs=10 --tweet_sarc_csv=sarc_thread.csv --tweet_non_sarc_csv=non_sarc_thread.csv --results_dir=results_eli_obl --class_=all
parser = ArgumentParser()
parser.add_argument("--model", dest="model", type=str)
parser.add_argument("--s_model", dest="s_model", default='sentence-transformers/all-distilroberta-v1', type=str)
parser.add_argument("--dim", dest="dim", default=768, type=int)
parser.add_argument("--num_epochs", dest="num_epochs", default=5, type=int)
parser.add_argument("--learning_rate", dest="learning_rate", default=1e-4, type=float)
parser.add_argument("--batch_size", dest="batch_size", default=32, type=int)
parser.add_argument("--loss_type", dest="loss_type", default='cross_entropy', type=str)
parser.add_argument("--tweet_sarc_csv", dest="tweet_sarc_csv", type=str)
parser.add_argument("--tweet_non_sarc_csv", dest="tweet_non_sarc_csv", type=str)
parser.add_argument("--results_dir", dest="results_dir", type=str)
parser.add_argument("--class_", dest="class_", type=str) # sarcastic (perceived / intended) or all
parser.add_argument("--log", dest="log", type=str) # log file name/destination

TIMESTAMP = get_current_timestamp()

if __name__ == '__main__':
    # ignore future warnings
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args = parser.parse_args()
    model_name = args.model
    s_model = args.s_model
    num_epoch = args.num_epochs
    learning_rate = args.learning_rate
    BATCH_SIZE = args.batch_size
    S_BERT_DIM = args.dim
    loss_type = args.loss_type
    sarc_csv = args.tweet_sarc_csv
    non_sarc_csv = args.tweet_non_sarc_csv
    results_dir = args.results_dir
    checkpoint_dir = os.path.join(results_dir, f'best_models/{TIMESTAMP}_best_model_sampled.pt')
    class_ = args.class_
    log_ = args.log

    handler = logging.FileHandler(f'{log_}_{TIMESTAMP}.log')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            handler,
            logging.StreamHandler()
        ]
    )

    logging.info("Device: {}".format(DEVICE))  # logging.info

    # creating a dataset but without author vocabulary and without removing punctuations
    if class_.lower() == 'sarcastic':
        dataset = SarcasticTweetData(sarc_csv=sarc_csv, punctuations=True, author_vocab=None)
    else:
        dataset = AuthorTweetDataset(author_vocab=None, sarc_csv=sarc_csv, non_sarc_csv=non_sarc_csv, dim=2,
                                     punctuations=True)

    # split data into train, validation, test
    tweet_ids = list(dataset.tweetIdToId.keys())
    labels = list(dataset.tweetIdToLabel.values())
    train_tweets, test_tweets, train_labels, test_labels = train_test_split(tweet_ids, labels, test_size=0.2,
                                                                            random_state=SEED)
    train_tweets, val_tweets, train_labels, val_labels = train_test_split(train_tweets, train_labels, test_size=0.15,
                                                                          random_state=SEED)

    # initialize raw data for train, validation, test
    raw_dataset = {'train': {'index': [], 'text': [], 'label': [], 'author_idx': []},
                   'val': {'index': [], 'text': [], 'label': [], 'author_idx': []},
                   'test': {'index': [], 'text': [], 'label': [], 'author_idx': []}}

    tokenizer = AutoTokenizer.from_pretrained(s_model)

    # initialize data
    initialize_data_oblivious(raw_dataset, dataset, train_tweets, 'train')
    initialize_data_oblivious(raw_dataset, dataset, val_tweets, 'val')
    initialize_data_oblivious(raw_dataset, dataset, test_tweets, 'test')

    # print size
    train_size_stats = "Training Size: {}, {} labels {}, {} labels {}".format(
        len(raw_dataset['train']['index']), dataset.encodingToLabel[1],
        raw_dataset['train']['label'].count(1), dataset.encodingToLabel[0],
        raw_dataset['train']['label'].count(0))
    logging.info(train_size_stats)

    val_size_stats = "Validation Size: {}, {} labels {}, {} labels {}".format(
        len(raw_dataset['val']['index']), dataset.encodingToLabel[1],
        raw_dataset['val']['label'].count(1), dataset.encodingToLabel[0],
        raw_dataset['val']['label'].count(0))
    logging.info(val_size_stats)

    test_size_stats = "Test Size: {}, {} labels {}, {} labels {}".format(
        len(raw_dataset['test']['index']), dataset.encodingToLabel[1],
        raw_dataset['test']['label'].count(1), dataset.encodingToLabel[0],
        raw_dataset['test']['label'].count(0))
    logging.info(test_size_stats)

    tokenizer.truncation_side = 'left'


    def tokenize_function(example):  # defining function to map to dataset
        return tokenizer(example["text"], truncation=True)


    ds = DatasetDict()
    for split, d in raw_dataset.items():  # split=('train', 'val', 'test');
        ds[split] = Dataset.from_dict(mapping=d, features=Features({'label': Value(dtype='int64'),
                                                                    'text': Value(dtype='string'),
                                                                    'index': Value(dtype='int64'),
                                                                    'author_idx': Value(dtype='int64')}))

    logging.info("Tokenizing the dataset.")
    tokenized_dataset = ds.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    logging.info("Training with the {} model".format(model_name))

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

    model = SentBertClassifier(sbert_dim=S_BERT_DIM, sbert_model=s_model, user_out_dim=BATCH_SIZE)
    model.to(DEVICE)

    # initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epoch * len(train_dataloader)
    samples_per_class_train = get_samples_per_class(tokenized_dataset["train"]['labels'])

    samples_per_class_train_tensor = torch.tensor(samples_per_class_train).float()
    loss_fn = nn.CrossEntropyLoss(samples_per_class_train_tensor.to(DEVICE))
    print(samples_per_class_train_tensor)

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
            output = model(batch)

            loss = loss_fn(output, labels)  # for cross entropy (with weights)
            train_loss.append(loss.item())
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        val_metric = evaluate_metric(eval_dataloader, model, None, False, dataset, None)
        val_metrics.append(val_metric)

        logging.info("Epoch {} **** Metrics validation: {}".format(epoch, val_metric))
        if val_metric['f1_weighted'] > best_f1:
            best_f1 = val_metric['f1_weighted']
            torch.save(model.state_dict(), checkpoint_dir)  # save model

    logging.info('Evaluating')
    model.load_state_dict(torch.load(checkpoint_dir))
    model.to(DEVICE)

    test_metrics = evaluate_metric(dataloader=test_dataloader, model=model, embedder='none', USE_AUTHORS=False,
                                   dataset=dataset, author_encoder=None, return_predictions=True)

    results = test_metrics.pop('results')
    logging.info('Test metrics: {}'.format(test_metrics))

    result_logs = dict()
    result_logs['id'] = TIMESTAMP
    result_logs['seed'] = SEED
    result_logs['model_name'] = 'S_bert with oblivious and eliciting'
    result_logs['train_stats'] = train_size_stats
    result_logs['val_stats'] = val_size_stats
    result_logs['test_stats'] = test_size_stats
    result_logs['epochs'] = num_epoch
    result_logs['batch_size'] = BATCH_SIZE
    result_logs['optimizer'] = optimizer.defaults
    result_logs["loss_type"] = loss_type
    result_logs['test_metrics'] = test_metrics
    result_logs['checkpoint_dir'] = checkpoint_dir
    result_logs['val_metrics'] = val_metrics
    result_logs['results'] = results

    res_file = os.path.join(results_dir, TIMESTAMP + ".json")
    with open(res_file, mode='w') as f:
        json.dump(result_logs, f, cls=NpEncoder, indent=2)
