import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModel
from sarcasm_detection.constants import *
from sarcasm_detection.util_functions import tfidf_embed_sentences


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class SentBertClassifier(nn.Module):
    def __init__(self, users_layer=False, user_dim=768,
                 num_outputs=2, sbert_dim=768,
                 sbert_model='sentence-transformers/all-MiniLM-L6-v2', user_out_dim=0):
        super().__init__()
        print("Initializing with user layer set to {}".format(users_layer))
        self.model = AutoModel.from_pretrained(sbert_model)
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(sbert_dim, sbert_dim // 2)
        self.users_layer = users_layer
        self.user_out_dim = user_out_dim

        if users_layer:
            self.user_linear1 = nn.Linear(user_dim, self.user_out_dim)
            print(user_dim, self.user_out_dim)
            comb_in_dim = sbert_dim // 2 + self.user_out_dim
            self.combine_linear = nn.Linear(comb_in_dim, comb_in_dim // 2)
            self.linear2 = nn.Linear(comb_in_dim // 2, num_outputs)
        else:
            self.linear2 = nn.Linear(sbert_dim // 2, num_outputs)

        self.relu = nn.ReLU()

    def forward(self, input, users_embeddings=None):
        bert_output = self.model(**input)
        pooled_output = mean_pooling(bert_output, input['attention_mask'])
        downsized_output = self.linear1(self.dropout(pooled_output))
        output = self.relu(downsized_output)

        if self.users_layer:
            users_output = self.dropout(self.relu(self.user_linear1(users_embeddings)))
            text_output = self.dropout(output)
            output = self.relu(self.combine_linear(torch.cat([text_output, users_output], dim=1)))

        output = self.linear2(self.dropout(output))
        return output

    def size(self):
        return sum(p.numel() for p in self.parameters())

    # Mean Pooling - Take attention mask into account for correct averaging



class SentBertClassifierAttribution(nn.Module):
    def __init__(self, user_dim=768, num_outputs=2, sbert_dim=768, sbert_model='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__()
        self.model = AutoModel.from_pretrained(sbert_model)
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(sbert_dim, sbert_dim // 2)
        self.user_linear1 = nn.Linear(user_dim, 2048)
        self.user_linear2 = nn.Linear(2048, sbert_dim)
        comb_in_dim = sbert_dim // 2 + sbert_dim
        self.combine_linear = nn.Linear(comb_in_dim, comb_in_dim // 2)
        self.linear2 = nn.Linear(comb_in_dim // 2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, input, users_embeddings):
        bert_output = self.model(**input)
        pooled_output = mean_pooling(bert_output, input['attention_mask'])
        downsized_output = self.linear1(self.dropout(pooled_output))
        output = self.relu(downsized_output)
        users_output = self.dropout(self.relu(self.user_linear1(users_embeddings)))
        users_output = self.dropout(self.relu(self.user_linear2(users_output)))
        text_output = self.dropout(output)
        output = self.relu(self.combine_linear(torch.cat([text_output, users_output], dim=1)))
        output = self.linear2(self.dropout(output))
        return output

    def size(self):
        return sum(p.numel() for p in self.parameters())

class MLPAttribution(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.relu(self.linear1(input))
        return self.linear2(F.dropout(output, p=0.2, training=self.training))
