import argparse
import os

from utils import Dataset

parser = argparse.ArgumentParser()

parser.add_argument("--load_paths", required=True, nargs='+', help="List of Load Paths")
parser.add_argument("--save_path", required=True, help="Path to save npz files")
parser.add_argument("--preprocess", default='mean', help="Preprocess, one of slice and mean")

args = parser.parse_args()

load_paths = args.load_paths
load_paths = ['../mri_data/IXI-Dataset/T1-Dataset', '../mri_data/IXI-Defaced/T1-Defaced']

preprocess = args.preprocess
save_path = args.save_path

dataset = Dataset(load_paths)
dataset.load_save_images(save_path, preprocess=args.preprocess)

print('Load And Save Done.')