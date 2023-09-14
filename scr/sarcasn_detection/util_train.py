from datasets import load_metric
from sklearn.metrics import f1_score, recall_score, precision_score
import torch
import pickle as pkl
from sarcasm_detection.constants import DEVICE
import evaluate


# class for the user embeddings
class AuthorsEmbedder:
    def __init__(self, embeddings_path, dim):
        self.authors_embeddings = pkl.load(open(embeddings_path, 'rb'))
        self.dim = dim

    def embed_author(self, author):
        return torch.tensor(self.authors_embeddings.get(author, torch.rand(self.dim)))


def evaluate_metric(dataloader, model, embedder, USE_AUTHORS, dataset, author_encoder,
             return_predictions=False, model_name='sbert'):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    recall = evaluate.load('recall')
    precision = evaluate.load('precision')

    model.eval()
    all_ids = ['tweet ids']
    all_pred = ['predictions']
    all_labels = ['gold labels']

    for batch in dataloader:
        tweet_index = batch.pop("index")
        author_idx = batch.pop("author_idx")
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.no_grad():
            if USE_AUTHORS and (author_encoder == 'average' or author_encoder.lower() == 'attribution'):
                authors_embeddings = torch.stack([embedder.embed_author(dataset.idToAuthor[dataset.authorToTweet[index.item()]][0]) for index in tweet_index]).to(DEVICE)
                logits = model(batch, authors_embeddings)
            elif model_name.lower() == 'lstm' and embedder.lower() =='glove':
                lengths = batch['text_len']
                lengths = lengths.int()
                lengths = lengths.cpu()
                logits = model(batch, lengths)
            else:
                logits = model(batch)

        predictions = torch.argmax(logits, dim=-1)
        accuracy_metric.add_batch(predictions=predictions, references=labels)
        f1_metric.add_batch(predictions=predictions, references=labels)
        all_pred.extend(predictions.cpu().numpy())  # add all predictions
        all_labels.extend(labels.cpu().numpy())  # add all labels
        all_ids.extend([dataset.tweetIdToId[idx] for idx in tweet_index.numpy()])  # adds enumeration of tweet id

    if return_predictions:
        return {'accuracy': accuracy_metric.compute()['accuracy'],
                'f1_weighted': f1_metric.compute(average='weighted')['f1'],
                'macro': f1_score(all_labels[1:], all_pred[1:], average='macro'),
                'micro': f1_score(all_labels[1:], all_pred[1:], average='micro'),
                'binary': f1_score(all_labels[1:], all_pred[1:], average='binary'),
                'recall_0': recall_score(all_labels[1:], all_pred[1:],average='binary', pos_label=0),
                'recall_1': recall_score(all_labels[1:], all_pred[1:],average='binary', pos_label=1),
                'recall_macro': recall_score(all_labels[1:], all_pred[1:],average='macro'),
                'recall_micro': recall_score(all_labels[1:], all_pred[1:],average='micro'),
                'precision_0': precision_score(all_labels[1:], all_pred[1:],average='binary', pos_label=0),
                'precision_1': precision_score(all_labels[1:], all_pred[1:],average='binary', pos_label=1),
                'precision_macro': precision_score(all_labels[1:], all_pred[1:],average='macro'),
                'precision_micro': precision_score(all_labels[1:], all_pred[1:],average='micro'),
                'results': list(zip(all_ids, all_pred, all_labels))}

    return {'accuracy': accuracy_metric.compute()['accuracy'],
            'f1_weighted': f1_metric.compute(average='weighted')['f1'],
            'macro': f1_score(all_labels[1:], all_pred[1:], average='macro'),
            'micro': f1_score(all_labels[1:], all_pred[1:], average='micro'),
            'binary': f1_score(all_labels[1:], all_pred[1:], average='binary'),
            'recall_0': recall_score(all_labels[1:], all_pred[1:],average='binary', pos_label=0),
            'recall_1': recall_score(all_labels[1:], all_pred[1:],average='binary', pos_label=1),
            'recall_macro': recall_score(all_labels[1:], all_pred[1:],average='macro'),
            'recall_micro': recall_score(all_labels[1:], all_pred[1:],average='micro'),
            'precision_0': precision_score(all_labels[1:], all_pred[1:],average='binary', pos_label=0),
            'precision_1': precision_score(all_labels[1:], all_pred[1:],average='binary', pos_label=1),
            'precision_macro': precision_score(all_labels[1:], all_pred[1:],average='macro'),
            'precision_micro': precision_score(all_labels[1:], all_pred[1:],average='micro')
            }

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
