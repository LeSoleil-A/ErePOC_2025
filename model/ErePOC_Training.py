import argparse

import copy
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader

# local dependencies 
from contrastive_func import Contrastive_loss
from utils import Net_embed_MLP

import warnings
warnings.filterwarnings('ignore')

# args
parser = argparse.ArgumentParser(description='KLCL')

parser.add_argument('--epochs', default=5000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batches', default=4096, type=int, metavar='N',
                    help='batch number per epoch')
parser.add_argument('--samples', default=128, type=int, metavar='N',
                    help='samples per batch')
parser.add_argument('--hidden', default=512, type=int, metavar='N',
                    help='hidden dimension size')
parser.add_argument('--outdim', default=256, type=int, metavar='N',
                    help='out dimension')

parser.add_argument('--disable-cuda', default=False, type=bool,
                    help='Disable CUDA')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')                  

parser.add_argument('--train-filepath', default="", type=str, metavar='N',
                    help='Trainint set file path')
parser.add_argument('--val-filepath', default="", type=str, metavar='N',
                    help='Validation set file path')

# Validation args
parser.add_argument('--valbatches', default=256, type=int, metavar='N',
                    help='batch number for validation')
parser.add_argument('--valsamples', default=128, type=int, metavar='N',
                    help='sampels per batch in validation')

# TorchDataset Class for Training
class MyDS(TorchDataset):
  def __init__(self, X: list, fingerprint: list):
    """Create a MyDS"""
    self._dat_list = []
    assert len(X) == len(fingerprint), 'Inconsistent length on `X` and `fingerprint`'
    for _ in range(len(X)):
      self._dat_list.append((X[_], fingerprint[_]))

  def __len__(self):
    return len(self._dat_list)

  def __getitem__(self, idx: int):
    return torch.FloatTensor(self._dat_list[idx][0]), torch.LongTensor(self._dat_list[idx][1])


def main():
  args = parser.parse_args()

  # Check CUDA
  if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
  else:
    raise SystemExit('CUDA is not available!')

  # Load Training Dataset
  csv_train = pd.read_csv(args.train_filepath)
  X_train = [np.asarray(mol[1:-1].split(","), dtype=np.float16) for mol in csv_train['esm2-1280'].to_list()]
  y_train = [np.asarray(fgp[1:-1].split(","), dtype=np.int32  ) for fgp in csv_train['Fingerprint'].to_list()]

  # Load Validation Dataset
  csv_val = pd.read_csv(args.val_filepath)
  X_val = [np.asarray(mol[1:-1].split(","), dtype=np.float16) for mol in csv_val['esm2-1280'].to_list()]
  y_val = [np.asarray(fgp[1:-1].split(","), dtype=np.int32  ) for fgp in csv_val['Fingerprint'].to_list()]

  print("Training Set Length: ", len(X_train), len(y_train), "Validation Set Length: ", len(X_val), len(y_val))
  
  train_loader = TorchDataLoader(MyDS(X=X_train, fingerprint=y_train), batch_size=args.samples, shuffle=True)
  val_loader = TorchDataLoader(MyDS(X=X_val, fingerprint=y_val), batch_size=args.valsamples, shuffle=True)

  # Training using deep contrastive learning  
  net = Net_embed_MLP(input_dim=X_train[0].shape[0],hidden_dim=args.hidden,out_dim=args.outdim, drop_prob=0.1)
  net.cuda()

  N_epoch = args.epochs
  optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
  # Load loss function
  criterion = Contrastive_loss(margin=0.2)

  # Initialize Variables for EarlyStopping
  best_loss = float('inf')
  best_model_weights = None
  best_epoch = -1
  patience = 300

  # Start Training
  for epo in range(N_epoch):
    net.train()
    batch_loss = 0

    for i_batch, tdata in enumerate(train_loader):
      tdata: tuple[torch.Tensor, torch.Tensor]
      x, y = tdata
      x = x.cuda()
      y = y.float().cuda()
      pred = net(x)
      loss = criterion(pred, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      batch_loss += float(loss)
    
    batch_loss = batch_loss / len(train_loader)
    print(f"epoch: {epo}  Loss:  {batch_loss:.5f}")

    # Validation
    net.eval()
    with torch.no_grad():  # Disable gradient calculation for validation
      val_loss = 0
      for i_batch, tdata in enumerate(val_loader):
        tdata: tuple[torch.Tensor, torch.Tensor]
        x, y = tdata
        x = x.cuda()
        y = y.float().cuda()
        pred_val = net(x)
        loss = criterion(pred_val, y)
        val_loss += float(loss)

      val_loss = val_loss / len(val_loader)

      # Early stopping
      if val_loss < best_loss:
        best_loss = val_loss
        best_model_weights = copy.deepcopy(net.state_dict()) # Deep copy here
        patience = 300 # Reset patience counter
        best_epoch = epo
      else:
        patience -= 1
        if patience == 0:
          break

      print(f"Val Loss:  {val_loss:.5f}  Val Best Loss:  {best_loss:.5f}")

  torch.save(best_model_weights, f"esm2-mlp-best-epoch-{str(best_epoch)}-newfps.pt")
  print("mlp best epo: ", best_epoch)


if __name__ == "__main__":
  main()
