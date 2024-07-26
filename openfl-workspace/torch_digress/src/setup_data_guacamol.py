import os
import os.path as osp
from torch_geometric.data import download_url
import hashlib
import random
from typing import List

TRAIN_HASH = '05ad85d871958a05c02ab51a4fde8530'
VALID_HASH = 'e53db4bff7dc4784123ae6df72e3b1f0'
TEST_HASH = '677b757ccec4809febd83850b43e1616'

def compare_hash(output_file: str, correct_hash: str) -> bool:
    """
    Computes the md5 hash of a SMILES file and check it against a given one
    Returns false if hashes are different
    """
    output_hash = hashlib.md5(open(output_file, 'rb').read()).hexdigest()
    if output_hash != correct_hash:
        print(f'{output_file} file has different hash, {output_hash}, than expected, {correct_hash}!')
        return False

    return True

def download(raw_dir: str):
    train_url = 'https://figshare.com/ndownloader/files/13612760'
    test_url = 'https://figshare.com/ndownloader/files/13612757'
    valid_url = 'https://figshare.com/ndownloader/files/13612766'

    train_path = download_url(train_url, raw_dir)
    os.rename(train_path, osp.join(raw_dir, 'guacamol_v1_train.smiles'))
    train_path = osp.join(raw_dir, 'guacamol_v1_train.smiles')

    test_path = download_url(test_url, raw_dir)
    os.rename(test_path, osp.join(raw_dir, 'guacamol_v1_test.smiles'))
    test_path = osp.join(raw_dir, 'guacamol_v1_test.smiles')

    valid_path = download_url(valid_url, raw_dir)
    os.rename(valid_path, osp.join(raw_dir, 'guacamol_v1_valid.smiles'))
    valid_path = osp.join(raw_dir, 'guacamol_v1_valid.smiles')

    # check the hashes
    # Check whether the md5-hashes of the generated smiles files match
    # the precomputed hashes, this ensures everyone works with the same splits.
    valid_hashes = [
        compare_hash(train_path, TRAIN_HASH),
        compare_hash(valid_path, VALID_HASH),
        compare_hash(test_path, TEST_HASH),
    ]

    if not all(valid_hashes):
        raise SystemExit('Invalid hashes for the dataset files')

    print('Dataset download successful. Hashes are correct.')


def split_smiles_files_into_folders(raw_dir: str, output_base_dir: str, num_splits: int, seed: int = 42):
    """
    Splits the SMILES files into multiple folders with random shuffling and prints the number of entries in each split.

    Parameters:
    - raw_dir: Directory where the raw SMILES files are located.
    - output_base_dir: Base directory where the split SMILES folders will be created.
    - num_splits: Number of splits/folders to create.
    - seed: Random seed for reproducibility.
    """
    # Set the random seed for reproducibility
    random.seed(seed)

    # Define the filenames
    filenames = ['guacamol_v1_train.smiles', 'guacamol_v1_valid.smiles', 'guacamol_v1_test.smiles']

    # Read and shuffle the contents of each file
    smiles_data = {}
    for filename in filenames:
        with open(osp.join(raw_dir, filename), 'r') as file:
            smiles_list = file.readlines()
            random.shuffle(smiles_list)
            smiles_data[filename] = smiles_list

    # Split the data and save into separate folders
    for i in range(num_splits):
        split_dir = osp.join(output_base_dir, f'split_{i+1}', 'raw')
        os.makedirs(split_dir, exist_ok=True)

        print(f"Creating split {i+1} in {split_dir} with the following number of entries:")
        for filename in filenames:
            # Determine the split for the current file
            split_size = len(smiles_data[filename]) // num_splits
            start_idx = i * split_size
            end_idx = None if i == num_splits - 1 else start_idx + split_size
            split_smiles = smiles_data[filename][start_idx:end_idx]

            # Save the split into the corresponding file in the split directory
            with open(osp.join(split_dir, filename), 'w') as file:
                file.writelines(split_smiles)

            # Print the number of entries for the current split
            num_entries = len(split_smiles)
            print(f"  {filename}: {num_entries} entries")


def count_smiles_in_splits(split_dir: str):
    """
    Counts the number of SMILES in each split directory.

    Parameters:
    - output_base_dir: Base directory where the split SMILES folders are located.
    - num_splits: Number of splits/folders to count.
    """
    # Define the filenames
    filenames = ['guacamol_v1_train.smiles', 'guacamol_v1_valid.smiles', 'guacamol_v1_test.smiles']
    
    for filename in filenames:
        file_path = osp.join(split_dir, filename)
        with open(file_path, 'r') as file:
            num_lines = sum(1 for line in file)
            print(f"  {filename}: {num_lines} entries")

# Example usage:
raw_dir = './data/raw'  # Directory containing the original .smiles files
output_base_dir = './data/'  # Base directory for the split folders
num_splits = 2  # Number of splits/folders to create

download(raw_dir)
split_smiles_files_into_folders(raw_dir, output_base_dir, num_splits)
# count_smiles_in_splits(raw_dir)
# count_smiles_in_splits('./data/splits/split_1')
# count_smiles_in_splits('./data/splits/split_1')