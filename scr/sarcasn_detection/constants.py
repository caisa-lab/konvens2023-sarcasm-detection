import torch

# defined constants
CUDA = 'cuda:2' # change if needed
CPU = 'cpu'
DEVICE = torch.device(CUDA if torch.cuda.is_available() else CPU)
SEED = 1234 # change, if needed
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"
