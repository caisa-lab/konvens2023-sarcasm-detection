from argparse import ArgumentParser
import pickle
from tqdm import tqdm
from sarcasm_detection.constants import *
import torch.nn.functional as F
from sarcasm_detection.models import MLPAttribution

# python user_attribution_extraction.py
# --model_file=/user_attribution/linear_layer_2023-01-21_17:31:30:294971.pt --embedding_type=prediction --embedding_file=text_embeddings_history.pkl --result_file=/embeddings/user_attribution.pkl

parser = ArgumentParser()
parser.add_argument("--model_file", dest="model_file",type=str)
parser.add_argument("--embedding_type", dest="embedding_type", type=str)  # ['distribution', 'prediction']
parser.add_argument("--embedding_file", dest="embedding_file", type=str)  # file with embeddings per user (user history)
parser.add_argument("--result_file", dest="result_file", type=str)
parser.add_argument("--data_", dest="data",dafaul='new', type=str)
parser.add_argument("--input_dim", dest="input_dim",dafaul=768, type=int)
parser.add_argument("--output_dim", dest="output_dim",dafaul=2048, type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    embedding_type = args.embedding_type

    # loading embeddings for each author (no need to load the data because the embedding has been created for all the authors of the dataset)
    with open(args.embedding_file, 'rb') as f:
        text_embeddings = pickle.load(f)

    print('Creating embeddings for {} users of the {} dataset.'.format(len(text_embeddings), args.data))

    # load model
    print('Loading model from: {}'.format(args.model_file))
    checkpoint = torch.load(args.model_file)
    model = MLPAttribution(args.input_dim, args.output_dim, checkpoint['num_output'])
    model.load_state_dict(checkpoint['model'])
    model.to(DEVICE)

    user_embeddings = {}
    DEBUG = True
    for author, tweets in tqdm(text_embeddings.items(), desc="Extracting author embeddings"):
        batch = [tensor for tensor in tweets] # no need to batch because each author has maximum tweets of 64
        batched_embeddings = torch.tensor(batch) # embedding file already contains a numpy array for each tweet in history
        if embedding_type == 'distribution':
            embeddings = []
        else:
            embeddings = torch.zeros(checkpoint['num_output'])

        size_of_embeddings = len(batch)
        with torch.no_grad():
            output = model(batched_embeddings.to(DEVICE))
            if embedding_type == 'distribution':
                output = F.normalize(output, p=2, dim=1)
                embedding = output.cpu().mean(axis=0).unsqueeze(0)
                embeddings.append(embedding)
            elif embedding_type == 'prediction':
                predictions = torch.argmax(output, dim=-1)
                for i in predictions:
                    embeddings[i] += 1
            else:
                raise Exception("Wrong embedding type")

        if embedding_type == 'distribution':
            if len(embeddings) > 1:
                embeddings = torch.cat(embeddings)
                user_embeddings[author] = embeddings.mean(axis=0).numpy()
            else:
                user_embeddings[author] = embeddings[0].squeeze().numpy()
        elif embedding_type == 'prediction':
            user_embeddings[author] = embeddings / size_of_embeddings
        else:
            raise Exception("Wrong embedding type")

        if DEBUG:
            print(batched_embeddings.size())
            print(user_embeddings[author], user_embeddings[author].shape)
            DEBUG = False

    result_file = args.result_file
    print("Saving embeddings to {}.".format(result_file))
    pickle.dump(user_embeddings, open(result_file, 'wb'))