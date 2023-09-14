from tqdm import tqdm
import pickle as pkl
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import numpy
from argparse import ArgumentParser
from sarcasm_detection.util_functions import *
from sarcasm_detection.constants import *
import logging
from sarcasm_detection.util_train import mean_pooling

# Example call
# python users_s_bert_embeddings.py --user_hist_file=user_vocab.pkl  --output_file=user_embeddings.pkl
parser = ArgumentParser()
parser.add_argument("--bert_model", dest="bert_model", default='sentence-transformers/all-distilroberta-v1', type=str)
parser.add_argument("--user_hist_file", dest="user_hist_file", type=str)
parser.add_argument("--output_file", dest="output_file", type=str)
parser.add_argument("--log", dest="log", type=str) # file/directory name where log file should be saved


TIMESTAMP = get_current_timestamp()

# run main and create user embeddings from the whole author history
if __name__ == '__main__':
    # parse user input
    args = parser.parse_args()
    user_hist = args.user_hist_file
    bert_mode = args.bert_model
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

    # read author vocab and get the corresponding dictionary
    with open(user_hist, 'rb') as f:
        author_hist = pickle.load(f)

    logging.info('We train the model on the {}.'.format(DEVICE))  # Check if CPU or GPU support

    # print length of author vocabulary (number of unique keys)
    logging.info('Creating user embeddings for {} users.'.format(len(list(author_hist.keys()))))
    logging.info("Using the following model: {}.".format(bert_mode))

    # Load tokenizer HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(bert_mode)

    model = AutoModel.from_pretrained(args.bert_model).to(DEVICE)
    user_embeddings = {} # start user embeddings

    # method to extract batches from author vocabulary
    def extract_batches(seq, batch_size=32):
        n = len(seq) // batch_size
        batches = []

        for i in range(n):
            batches.append(seq[i * batch_size:(i + 1) * batch_size])
        if len(seq) % batch_size != 0:
            batches.append(seq[n * batch_size:])
        return batches

    # loop through the whole user hist dictionary
    DEBUG = True
    for author, texts in tqdm(author_hist.items(), desc="Embedding authors"):
        # clean the text. Default no emojis and usernames (maybe useful for that task)
        processed_texts = [process_tweet(text) for text in texts] # lists of sentences in one list
        batches_text = extract_batches(processed_texts, 64)  # Tokenize sentences
        embeddings = []
        encoded_inputs = [tokenizer(processed_texts, padding=True, truncation=True, return_tensors='pt') for
                          processed_texts in batches_text]

        for encoded_input in encoded_inputs:
            with torch.no_grad():
                encoded_input = {k: v.to(DEVICE) for k, v in encoded_input.items()} # Compute token embeddings
                model_output = model(**encoded_input)
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']) # Perform model encode
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1) # Normalize embeddings
                average = sentence_embeddings.cpu().mean(axis=0)
                embeddings.append(average.unsqueeze(0))

        if len(embeddings) > 1:
            embedding = torch.cat(embeddings)
            user_embeddings[author] = embedding.mean(axis=0).numpy()
        else:
            user_embeddings[author] = embeddings[0].squeeze().numpy()

        if DEBUG:
            print(user_embeddings[author], user_embeddings[author].shape)
            DEBUG = False

    print("Saving embeddings")
    pkl.dump(user_embeddings, open(args.output_file, 'wb'))
