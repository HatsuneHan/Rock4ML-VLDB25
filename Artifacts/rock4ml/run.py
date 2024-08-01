import argparse
import os

import rock4ml
from rock4ml import Rock4ML
import pandas as pd
import time

def initParser():
  parser = argparse.ArgumentParser(description='Clean the dataset')
  parser.add_argument('-r', '--root_dir', type=str, help='The root directory', required=True)
  parser.add_argument('-d', '--dataset', type=str, help='The dataset name', required=True)
  parser.add_argument('-c', '--cr_method', type=str,  help='CR method', required=True)
  parser.add_argument('-m', '--model_name', type=str,  help='Model Name', required=True)

  parser.add_argument('-e', '--epoch', type=int,  help='epoch for inner-model training', default=30)
  parser.add_argument('-b', '--batch_size', type=int,  help='batch size for training', default=16)
  parser.add_argument('-s', '--random_state', type=int,  help='Random Seed', default=42)

  parser.add_argument('-pi', '--temperature', type=float,  help='coerset error signal temperature', default=30)
  
  parser.add_argument('-kr', '--kinf_ratio', type=float,  help='ratio for topk influential tuples', default=0.02)

  parser.add_argument('-kg', '--topk_ga', type=float,  help='topk attr for ga', default=0)
  parser.add_argument('-gi', '--maxiter_ga', type=int,  help='Max iter in GA', default=3)
  parser.add_argument('-ie', '--inc_epoch', type=int,  help='Epoch for Incremental learning', default=0)

  # current unavailable
  parser.add_argument('-rp', '--rock_port', type=int,  help='port for rock', default=19123)


  return parser

if __name__ == "__main__":
  args = initParser().parse_args()

  root_dir = args.root_dir
  dataset_name = args.dataset
  dataset_label_dict = {"adult": "income", "nursery": "final evaluation", 
                        "default": "default.payment.next.month", "Bank": "y",
                        "german": "class", "road_safety": "SexofDriver"}

  base_dir = os.path.join(root_dir, dataset_name, 'exp')

  X_train_dirty = pd.read_csv(os.path.join(base_dir, 'X_train_dirty.csv'), na_values=['?'])
  X_val_dirty = pd.read_csv(os.path.join(base_dir, 'X_val_dirty.csv'), na_values=['?'])
  X_train_dirty = pd.concat([X_train_dirty, X_val_dirty], axis=0).reset_index(drop=True)

  X_train_clean = pd.read_csv(os.path.join(base_dir, 'X_train_clean.csv'), na_values=['?'])
  X_val_clean = pd.read_csv(os.path.join(base_dir, 'X_val.csv'), na_values=['?'])
  X_train_clean = pd.concat([X_train_clean, X_val_clean], axis=0).reset_index(drop=True)

  y_train = pd.read_csv(os.path.join(base_dir, 'y_train.csv'), na_values=['?'])
  y_val = pd.read_csv(os.path.join(base_dir, 'y_val.csv'), na_values=['?'])
  y_train = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

  X_test = pd.read_csv(os.path.join(base_dir, 'X_test.csv'), na_values=['?'])
  y_test = pd.read_csv(os.path.join(base_dir, 'y_test.csv'), na_values=['?'])

  dataset_params = {"name": dataset_name,
                    "label_column": dataset_label_dict[dataset_name],
                    "dataset_rootdir": root_dir} # some model need the name of the label column

  rock4ml_params = {"model_name": args.model_name,
                    "cr_method": args.cr_method,
                    "maxiter_ga": args.maxiter_ga,
                    "no_epoch": args.epoch,
                    "batch_size": args.batch_size,
                    "random_state": args.random_state,
                    "temperature": args.temperature,
                    "kinf_ratio": args.kinf_ratio,
                    "rock_port": args.rock_port,
                    "topk_ga": args.topk_ga,
                    "inc_epoch": args.inc_epoch}

  start_time = time.time()
  rock4ml = Rock4ML(dataset_params, rock4ml_params)
  rock4ml.fit(X_train_dirty, y_train, X_test, y_test, X_train_clean)
  print("Time taken for", args.cr_method, ":", time.time() - start_time)