import os
from argparse import ArgumentParser
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import evaluate
from tqdm import tqdm
from sarcasm_detection.dataset import AuthorTweetDataset, SarcasticTweetData
from sarcasm_detection.models import MLPAttribution
from sarcasm_detection.util_functions import get_current_timestamp, process_tweet
from sarcasm_detection.constants import DEVICE, SEED

# example call train_user_attribution.py
#--sarc_csv=sarc_thread.csv --non_sarc_csv=non_sarc_thread.csv --text_embeddings=text_embeddings_dataset.pkl --result_dir=user_attribution --class_=all

parser = ArgumentParser()
parser.add_argument("--sarc_csv", dest="sarc_csv", type=str) # file that only contains tweets for which user history exists
parser.add_argument("--non_sarc_csv", dest="non_sarc_csv", type=str) # file that only contains tweets for which user history exists
parser.add_argument("--text_embeddings", dest="text_embeddings", type=str)
parser.add_argument("--result_dir", dest="result_dir", type=str)
parser.add_argument("--class_", dest="class_", type=str)
parser.add_argument("--data_", dest="data",dafaul='new', type=str)
parser.add_argument("--input_dim", dest="input_dim",dafaul=768, type=int)
parser.add_argument("--output_dim", dest="output_dim",dafaul=2048, type=int)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

TIMESTAMP = get_current_timestamp()


def evaluate_mlp(model, dataloader):
    accuracy_metric = evaluate.load("accuracy")
    f1_macro = evaluate.load("f1")
    f1_weighted = evaluate.load("f1")

    model.eval()
    for batch in dataloader:
        with torch.no_grad():
            input = batch[0]
            labels = batch[1].to(DEVICE)
            embedding = embedding_layer(input).to(DEVICE)
            logits = model(embedding)

        predictions = torch.argmax(logits, dim=-1)
        accuracy_metric.add_batch(predictions=predictions, references=labels)
        f1_macro.add_batch(predictions=predictions, references=labels)
        f1_weighted.add_batch(predictions=predictions, references=labels)

    return {'accuracy': accuracy_metric.compute()['accuracy'],
            'f1_score_macro': f1_macro.compute(average="macro")['f1'],
            'f1_score_weighted': f1_weighted.compute(average="weighted")['f1']}

if __name__ == '__main__':
    args = parser.parse_args()
    # load text embeddings and dataset
    if args.class_ == 'sarcastic':
        dataset = SarcasticTweetData(sarc_csv=args.sarc_csv, punctuations=False) # need to be changed to true if you don't have previous cleaned data
    elif args.class_ == 'all':
        dataset = AuthorTweetDataset(None, args.sarc_csv, args.non_sarc_csv, 2, punctuations=False) # need to be changed to true if you don't have previous cleaned data
    else:
        print("The given class is not supported.")
        dataset = None
    with open(args.text_embeddings, 'rb') as f:
        text_embeddings = pickle.load(f)




    # split the dataset
    tweet_ids = list(dataset.tweetIdToId.keys())  # get all tweet ids
    # discard tweets of users with less than 5 tokens
    labels = list(dataset.tweetIdToLabel.values())  # get all labels to tweet ids
    train_tweets, test_tweets, train_labels, test_labels = train_test_split(tweet_ids, labels, test_size=0.2,
                                                                            random_state=SEED)

    train_tweets, val_tweets, train_labels, val_labels = train_test_split(train_tweets, train_labels, test_size=0.15,
                                                                          random_state=SEED)

    # get all unique authors
    authors = set()
    for tweet in train_tweets: # getting all unique training authors
        # tweetIdToId gives the id string to the enumerated index
        # tweetToAuthorString gives the author id string to the tweet id string
        authors.add(dataset.tweetToAuthorString[dataset.tweetIdToId[tweet]])

    # creating dictionary with the enumeration as value and author string as key
    authorToId = dict()
    for i, num in enumerate(authors):
        authorToId[num] = i

    # dictionary with tweet id: author id (only for train)
    tweetToAuthorLabel = dict()
    for tweet in train_tweets:
        string_tweet_id = dataset.tweetIdToId[tweet]
        string_author_id = dataset.tweetToAuthorString[string_tweet_id]
        if string_author_id in authorToId:
            tweetToAuthorLabel[string_tweet_id] = authorToId[string_author_id] # enumeration of the author

    embeddings = []
    tweetToTrainId = dict()
    all_tweet_str_ids = []
    all_author_labels = []

    for tweet_id_str in tqdm(tweetToAuthorLabel.keys()):
        if str(tweet_id_str) in text_embeddings: # str because, saved as str value
            tweetToTrainId[tweet_id_str] = len(tweetToTrainId) # en
            all_tweet_str_ids.append(tweetToTrainId[tweet_id_str])
            all_author_labels.append(tweetToAuthorLabel[tweet_id_str])
            embeddings.append(torch.tensor(text_embeddings[str(tweet_id_str)]).unsqueeze(0))

    print('Loading the data into dataloaders...')
    temp = torch.cat(embeddings, dim=0) # temporary tensor containing all embeddings
    embedding_layer = nn.Embedding.from_pretrained(temp)

    print('Number of tweets in the training dataset: {}'.format(len(all_tweet_str_ids)))

    # preparing train and validation set out of the train tweets
    train_tweet_mlp, val_tweet_mlp, train_labels_mlp, val_labels_mlp = train_test_split(all_tweet_str_ids, all_author_labels, test_size=0.2,
                                                                              random_state=SEED)

    train_dataloader = DataLoader([(train_tweet_mlp[i], train_labels_mlp[i]) for i in range(len(train_tweet_mlp))],
                                  batch_size=64, shuffle=True)
    val_dataloader = DataLoader([(val_tweet_mlp[i], val_labels_mlp[i]) for i in range(len(val_tweet_mlp))], batch_size=64)

    output_classes = len(authorToId)
    print("Number of output classes {}".format(output_classes))

    model = MLPAttribution(args.input_dim, args.output_dim, output_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    checkpoint_dir = os.path.join(f'{args.result_dir}/{args.class_}_{args.data}_linear_layer_{TIMESTAMP}.pt')
    num_epochs = 100
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    best_accuracy = 0
    best_f1 = 0

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            input = batch[0]
            labels = batch[1].to(DEVICE)
            embedding = embedding_layer(input).to(DEVICE)
            output = model(embedding)
            loss = loss_fn(input=output, target=labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        val_metric = evaluate_mlp(model, val_dataloader)

        print("Epoch {} **** Metrics validation: {}".format(epoch, val_metric))
        if val_metric['f1_score_weighted'] > best_f1:
            best_f1 = val_metric['f1_score_weighted']
            torch.save({'num_output': output_classes, 'model': model.state_dict()}, checkpoint_dir)

